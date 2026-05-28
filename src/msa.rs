//! Multiple Sequence Alignment (MSA) I/O.

use crate::alphabet::{Alphabet, Dsq, DSQ_ILLEGAL};
use crate::errors::{HmmerError, HmmerResult};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

const MAX_TEXT_MSA_BYTES: usize = 512 * 1024 * 1024;

fn read_text_msa_to_string<R: Read>(reader: &mut R) -> HmmerResult<String> {
    read_text_msa_to_string_with_limit(reader, MAX_TEXT_MSA_BYTES)
}

fn read_text_msa_to_string_with_limit<R: Read>(
    reader: &mut R,
    limit: usize,
) -> HmmerResult<String> {
    let mut bytes = Vec::new();
    let max_read = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    reader
        .take(max_read)
        .read_to_end(&mut bytes)
        .map_err(HmmerError::Io)?;
    if bytes.len() > limit {
        return Err(HmmerError::Format(format!(
            "MSA input exceeds {} bytes",
            limit
        )));
    }
    String::from_utf8(bytes)
        .map_err(|e| HmmerError::Format(format!("invalid UTF-8 in MSA input: {e}")))
}

/// Stockholm model-specific bit score cutoffs from `#=GF GA/TC/NC`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StockholmCutoffs {
    pub ga: Option<[f32; 2]>,
    pub tc: Option<[f32; 2]>,
    pub nc: Option<[f32; 2]>,
}

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
    /// Per-sequence relative weights (`#=GS <seq> WT`) if supplied.
    pub weights: Option<Vec<f64>>,
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
    /// Model mask annotation (#=GC MM)
    pub mm: Option<Vec<u8>>,
    /// Consensus secondary structure annotation (#=GC SS_cons)
    pub ss_cons: Option<Vec<u8>>,
    /// Consensus surface accessibility annotation (#=GC SA_cons)
    pub sa_cons: Option<Vec<u8>>,
    /// Consensus posterior probability annotation (#=GC PP_cons)
    pub pp_cons: Option<Vec<u8>>,
}

/// A parsed Stockholm alignment plus the original body lines needed for
/// metadata-preserving round trips.
#[derive(Debug, Clone)]
pub struct StockholmMsa {
    pub msa: Msa,
    pub cutoffs: StockholmCutoffs,
    pub body_lines: Vec<String>,
}

impl Msa {
    /// Validate that every stored alignment character can be digitized by
    /// `abc`. Gap characters are accepted explicitly because aligned MSA text
    /// maps them to the alphabet gap code before normal symbol digitization.
    pub fn validate_digitizable(&self, abc: &Alphabet) -> HmmerResult<()> {
        for (idx, seq) in self.aseq.iter().enumerate() {
            for &ch in seq {
                if ch == b'-' || ch == b'.' {
                    continue;
                }
                let code = abc.digitize_symbol(ch);
                if code == DSQ_ILLEGAL {
                    let display = if ch.is_ascii_graphic() || ch == b' ' {
                        (ch as char).to_string()
                    } else {
                        format!("\\x{ch:02x}")
                    };
                    let name = self.sqname.get(idx).map(String::as_str).unwrap_or("?");
                    return Err(HmmerError::Format(format!(
                        "Stockholm sequence {name} contains illegal symbol '{display}'"
                    )));
                }
            }
        }
        Ok(())
    }

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
    Ok(read_stockholm_preserved(path)?
        .into_iter()
        .map(|stockholm| stockholm.msa)
        .collect())
}

/// Read every Stockholm alignment in `path`, preserving original body lines
/// alongside parsed MSA fields.
pub fn read_stockholm_preserved(path: &Path) -> HmmerResult<Vec<StockholmMsa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader = BufReader::new(file);
    read_stockholm_preserved_from_reader(reader)
}

/// Read one aligned FASTA/AFA alignment from `path`.
pub fn read_afa(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_afa_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one aligned FASTA/AFA alignment from an open reader.
pub fn read_afa_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_afa_alignment(&text, alignment_name)?])
}

/// Read one A2M alignment from `path`.
pub fn read_a2m(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_a2m_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one A2M alignment from an open reader.
///
/// A2M marks consensus columns with uppercase residues or `-`; lowercase
/// residues are inserts, and `.` insert-column placeholders are ignored.
/// Rows may therefore have unequal raw lengths as long as they contain the
/// same number of consensus columns.
pub fn read_a2m_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_a2m_alignment(&text, alignment_name)?])
}

/// Read one PSIBLAST alignment from `path`.
pub fn read_psiblast(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_psiblast_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one PSIBLAST alignment from an open reader.
pub fn read_psiblast_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_psiblast_alignment(&text, alignment_name)?])
}

/// Read one CLUSTAL/CLUSTAL-like alignment from `path`.
pub fn read_clustal(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_clustal_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one CLUSTAL/CLUSTAL-like alignment from an open reader.
pub fn read_clustal_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_clustal_alignment(&text, alignment_name)?])
}

/// Read one SELEX alignment from `path`.
pub fn read_selex(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_selex_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one SELEX alignment from an open reader.
pub fn read_selex_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_selex_alignment(&text, alignment_name)?])
}

/// Read one interleaved PHYLIP alignment from `path`.
pub fn read_phylip(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_phylip_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one interleaved PHYLIP alignment from an open reader.
pub fn read_phylip_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_interleaved_phylip_alignment(
        &text,
        alignment_name,
    )?])
}

/// Read one sequential PHYLIP alignment from `path`.
pub fn read_phylips(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_phylips_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one sequential PHYLIP alignment from an open reader.
pub fn read_phylips_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_sequential_phylip_alignment(
        &text,
        alignment_name,
    )?])
}

fn parse_afa_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut sqname = Vec::new();
    let mut sqdesc = Vec::new();
    let mut aseq: Vec<Vec<u8>> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_desc = String::new();
    let mut current_seq = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        if let Some(header) = trimmed.strip_prefix('>') {
            if header.trim().is_empty() {
                return Err(HmmerError::Format(format!(
                    "aligned FASTA record at line {} has an empty name",
                    line_idx + 1
                )));
            }
            flush_afa_record(
                &mut current_name,
                &mut current_desc,
                &mut current_seq,
                &mut sqname,
                &mut sqdesc,
                &mut aseq,
            )?;
            let mut fields = header.trim().splitn(2, char::is_whitespace);
            current_name = fields.next().map(str::to_string);
            current_desc = fields.next().unwrap_or("").trim().to_string();
            continue;
        }
        if current_name.is_none() {
            return Err(HmmerError::Format(format!(
                "aligned FASTA sequence data before first header at line {}",
                line_idx + 1
            )));
        }
        for raw in trimmed.bytes() {
            if !raw.is_ascii_whitespace() {
                current_seq.push(raw);
            }
        }
    }

    flush_afa_record(
        &mut current_name,
        &mut current_desc,
        &mut current_seq,
        &mut sqname,
        &mut sqdesc,
        &mut aseq,
    )?;
    if sqname.is_empty() {
        return Err(HmmerError::Format(
            "aligned FASTA alignment contains no records".to_string(),
        ));
    }
    let alen = aseq[0].len();
    for (name, seq) in sqname.iter().zip(&aseq) {
        if seq.len() != alen {
            return Err(HmmerError::Format(format!(
                "aligned FASTA sequence {name} has aligned length {}, expected {alen}",
                seq.len()
            )));
        }
    }
    let nseq = sqname.len();
    Ok(Msa {
        name: alignment_name,
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqdesc,
        weights: None,
        aseq,
        pp: vec![None; nseq],
        nseq,
        alen,
        rf: None,
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: None,
    })
}

struct A2mRecord {
    name: String,
    desc: String,
    inserts: Vec<Vec<u8>>,
    consensus: Vec<u8>,
}

fn parse_a2m_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut records = Vec::<A2mRecord>::new();
    let mut current_name: Option<String> = None;
    let mut current_desc = String::new();
    let mut current_raw = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        if let Some(header) = trimmed.strip_prefix('>') {
            if header.trim().is_empty() {
                return Err(HmmerError::Format(format!(
                    "A2M record at line {} has an empty name",
                    line_idx + 1
                )));
            }
            flush_a2m_record(
                &mut current_name,
                &mut current_desc,
                &mut current_raw,
                &mut records,
            )?;
            let mut fields = header.trim().splitn(2, char::is_whitespace);
            current_name = fields.next().map(str::to_string);
            current_desc = fields.next().unwrap_or("").trim().to_string();
            continue;
        }
        if current_name.is_none() {
            return Err(HmmerError::Format(format!(
                "A2M sequence data before first header at line {}",
                line_idx + 1
            )));
        }
        for raw in trimmed.bytes() {
            if !raw.is_ascii_whitespace() {
                current_raw.push(raw);
            }
        }
    }

    flush_a2m_record(
        &mut current_name,
        &mut current_desc,
        &mut current_raw,
        &mut records,
    )?;
    if records.is_empty() {
        return Err(HmmerError::Format(
            "A2M alignment contains no records".to_string(),
        ));
    }

    let ncons = records[0].consensus.len();
    if ncons == 0 {
        return Err(HmmerError::Format(
            "A2M alignment contains no consensus columns".to_string(),
        ));
    }
    for record in &records {
        if record.consensus.len() != ncons {
            return Err(HmmerError::Format(format!(
                "A2M sequence {} has {} consensus columns, expected {ncons}",
                record.name,
                record.consensus.len()
            )));
        }
    }

    let mut insert_widths = vec![0usize; ncons + 1];
    for record in &records {
        for (idx, insert) in record.inserts.iter().enumerate() {
            insert_widths[idx] = insert_widths[idx].max(insert.len());
        }
    }

    let alen = ncons + insert_widths.iter().sum::<usize>();
    let mut rf = Vec::with_capacity(alen);
    for (idx, &width) in insert_widths.iter().enumerate().take(ncons + 1) {
        rf.extend(std::iter::repeat_n(b'.', width));
        if idx < ncons {
            rf.push(b'x');
        }
    }

    let mut sqname = Vec::with_capacity(records.len());
    let mut sqdesc = Vec::with_capacity(records.len());
    let mut aseq = Vec::with_capacity(records.len());
    for record in records {
        let mut row = Vec::with_capacity(alen);
        for (idx, &width) in insert_widths.iter().enumerate().take(ncons + 1) {
            let insert = &record.inserts[idx];
            row.extend_from_slice(insert);
            // Insert (non-consensus) gap columns use '.' in A2M; C
            // `esl_msafile_a2m.c` emits '.' here (consensus gaps use '-').
            row.extend(std::iter::repeat_n(b'.', width - insert.len()));
            if idx < ncons {
                row.push(record.consensus[idx]);
            }
        }
        sqname.push(record.name);
        sqdesc.push(record.desc);
        aseq.push(row);
    }
    let nseq = sqname.len();
    Ok(Msa {
        name: alignment_name,
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqdesc,
        weights: None,
        aseq,
        pp: vec![None; nseq],
        nseq,
        alen,
        rf: Some(rf),
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: None,
    })
}

fn parse_psiblast_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut sqname = Vec::new();
    let mut sqdesc = Vec::new();
    let mut aseq = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(seq) = fields.next() else {
            return Err(HmmerError::Format(format!(
                "PSIBLAST alignment row at line {} is missing sequence data",
                line_idx + 1
            )));
        };
        if fields.next().is_some() {
            return Err(HmmerError::Format(format!(
                "PSIBLAST alignment row at line {} has unexpected trailing fields",
                line_idx + 1
            )));
        }
        sqname.push(name.to_string());
        sqdesc.push(String::new());
        aseq.push(seq.as_bytes().to_vec());
    }

    msa_from_rows("PSIBLAST", alignment_name, sqname, sqdesc, aseq)
}

fn parse_clustal_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut lines = text.lines();
    let Some(header) = lines.find(|line| !line.trim().is_empty()) else {
        return Err(HmmerError::Format(
            "CLUSTAL alignment contains no records".to_string(),
        ));
    };
    let header = header.trim_start();
    if !header.starts_with("CLUSTAL") && !header.starts_with("MUSCLE") {
        return Err(HmmerError::Format(
            "CLUSTAL alignment does not start with a CLUSTAL/MUSCLE header".to_string(),
        ));
    }

    let mut order = Vec::<String>::new();
    let mut rows = HashMap::<String, Vec<u8>>::new();
    for (line_idx, line) in lines.enumerate() {
        if line.trim().is_empty() || line.starts_with(char::is_whitespace) {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(seq) = fields.next() else {
            return Err(HmmerError::Format(format!(
                "CLUSTAL alignment row at line {} is missing sequence data",
                line_idx + 2
            )));
        };
        if !rows.contains_key(name) {
            order.push(name.to_string());
        }
        rows.entry(name.to_string())
            .or_default()
            .extend_from_slice(seq.as_bytes());
    }

    let mut sqname = Vec::with_capacity(order.len());
    let mut sqdesc = Vec::with_capacity(order.len());
    let mut aseq = Vec::with_capacity(order.len());
    for name in order {
        let seq = rows
            .remove(&name)
            .ok_or_else(|| HmmerError::Format(format!("missing CLUSTAL row for {name}")))?;
        sqname.push(name);
        sqdesc.push(String::new());
        aseq.push(seq);
    }
    msa_from_rows("CLUSTAL", alignment_name, sqname, sqdesc, aseq)
}

fn parse_selex_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut order = Vec::<String>::new();
    let mut rows = HashMap::<String, Vec<u8>>::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('%')
            || trimmed.starts_with("//")
        {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(seq) = fields.next() else {
            return Err(HmmerError::Format(format!(
                "SELEX alignment row at line {} is missing sequence data",
                line_idx + 1
            )));
        };
        let mut chunk = seq.as_bytes().to_vec();
        for field in fields {
            chunk.extend_from_slice(field.as_bytes());
        }
        if !rows.contains_key(name) {
            order.push(name.to_string());
        }
        rows.entry(name.to_string()).or_default().extend(chunk);
    }

    let mut sqname = Vec::with_capacity(order.len());
    let mut sqdesc = Vec::with_capacity(order.len());
    let mut aseq = Vec::with_capacity(order.len());
    for name in order {
        let seq = rows
            .remove(&name)
            .ok_or_else(|| HmmerError::Format(format!("missing SELEX row for {name}")))?;
        sqname.push(name);
        sqdesc.push(String::new());
        aseq.push(seq);
    }
    msa_from_rows("SELEX", alignment_name, sqname, sqdesc, aseq)
}

fn parse_interleaved_phylip_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let (nseq, alen, records) = parse_phylip_lines(text)?;
    let mut sqname = Vec::with_capacity(nseq);
    let mut sqdesc = Vec::with_capacity(nseq);
    let mut aseq = vec![Vec::<u8>::new(); nseq];

    for (row, (line_idx, line)) in records.into_iter().enumerate() {
        if row < nseq {
            let (name, chunk) = parse_named_phylip_row(line, line_idx)?;
            sqname.push(name.to_string());
            sqdesc.push(String::new());
            aseq[row].extend_from_slice(chunk.as_bytes());
        } else {
            let seq_idx = row % nseq;
            let chunk = parse_interleaved_phylip_continuation(line, &sqname[seq_idx]);
            if !chunk.is_empty() {
                aseq[seq_idx].extend_from_slice(chunk.as_bytes());
            }
        }
    }

    if sqname.len() != nseq {
        return Err(HmmerError::Format(format!(
            "PHYLIP alignment declares {nseq} sequences but contains {}",
            sqname.len()
        )));
    }
    msa_from_rows_with_alen("PHYLIP", alignment_name, sqname, sqdesc, aseq, alen)
}

fn parse_sequential_phylip_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let (nseq, alen, records) = parse_phylip_lines(text)?;
    let mut iter = records.into_iter();
    let mut sqname = Vec::with_capacity(nseq);
    let mut sqdesc = Vec::with_capacity(nseq);
    let mut aseq = Vec::with_capacity(nseq);

    for seq_idx in 0..nseq {
        let Some((line_idx, line)) = iter.next() else {
            return Err(HmmerError::Format(format!(
                "PHYLIPS alignment declares {nseq} sequences but contains {seq_idx}"
            )));
        };
        let (name, first_chunk) = parse_named_phylip_row(line, line_idx)?;
        let mut seq = first_chunk.as_bytes().to_vec();
        while seq.len() < alen {
            let Some((_cont_idx, continuation)) = iter.next() else {
                return Err(HmmerError::Format(format!(
                    "PHYLIPS sequence {name} has aligned length {}, expected {alen}",
                    seq.len()
                )));
            };
            seq.extend_from_slice(strip_phylip_sequence_spaces(continuation).as_bytes());
        }
        sqname.push(name.to_string());
        sqdesc.push(String::new());
        aseq.push(seq);
    }

    if let Some((line_idx, _)) = iter.next() {
        return Err(HmmerError::Format(format!(
            "PHYLIPS alignment has extra data after {nseq} sequences at line {}",
            line_idx + 1
        )));
    }
    msa_from_rows_with_alen("PHYLIPS", alignment_name, sqname, sqdesc, aseq, alen)
}

type PhylipRows<'a> = Vec<(usize, &'a str)>;

fn parse_phylip_lines(text: &str) -> HmmerResult<(usize, usize, PhylipRows<'_>)> {
    let mut lines = text.lines().enumerate().filter_map(|(idx, line)| {
        let trimmed = line.trim();
        (!trimmed.is_empty()).then_some((idx, trimmed))
    });
    let Some((header_idx, header)) = lines.next() else {
        return Err(HmmerError::Format(
            "PHYLIP alignment contains no records".to_string(),
        ));
    };
    let mut fields = header.split_whitespace();
    let nseq = fields
        .next()
        .ok_or_else(|| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} is missing nseq",
                header_idx + 1
            ))
        })?
        .parse::<usize>()
        .map_err(|_| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} has invalid nseq",
                header_idx + 1
            ))
        })?;
    let alen = fields
        .next()
        .ok_or_else(|| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} is missing alen",
                header_idx + 1
            ))
        })?
        .parse::<usize>()
        .map_err(|_| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} has invalid alen",
                header_idx + 1
            ))
        })?;
    if nseq == 0 || alen == 0 {
        return Err(HmmerError::Format(
            "PHYLIP header declares zero sequences or columns".to_string(),
        ));
    }
    Ok((nseq, alen, lines.collect()))
}

fn parse_named_phylip_row(line: &str, line_idx: usize) -> HmmerResult<(&str, String)> {
    let mut fields = line.split_whitespace();
    let name = fields.next().ok_or_else(|| {
        HmmerError::Format(format!("PHYLIP row at line {} has no name", line_idx + 1))
    })?;
    let chunk: String = fields.collect();
    if chunk.is_empty() {
        return Err(HmmerError::Format(format!(
            "PHYLIP row at line {} is missing sequence data",
            line_idx + 1
        )));
    }
    Ok((name, chunk))
}

fn parse_interleaved_phylip_continuation(line: &str, expected_name: &str) -> String {
    let mut fields = line.split_whitespace();
    if fields.next() == Some(expected_name) {
        fields.collect()
    } else {
        strip_phylip_sequence_spaces(line)
    }
}

fn strip_phylip_sequence_spaces(line: &str) -> String {
    line.split_whitespace().collect()
}

fn msa_from_rows_with_alen(
    format_name: &str,
    alignment_name: String,
    sqname: Vec<String>,
    sqdesc: Vec<String>,
    aseq: Vec<Vec<u8>>,
    declared_alen: usize,
) -> HmmerResult<Msa> {
    let msa = msa_from_rows(format_name, alignment_name, sqname, sqdesc, aseq)?;
    if msa.alen != declared_alen {
        return Err(HmmerError::Format(format!(
            "{format_name} alignment declares aligned length {declared_alen} but parsed {}",
            msa.alen
        )));
    }
    Ok(msa)
}

fn msa_from_rows(
    format_name: &str,
    alignment_name: String,
    sqname: Vec<String>,
    sqdesc: Vec<String>,
    aseq: Vec<Vec<u8>>,
) -> HmmerResult<Msa> {
    if sqname.is_empty() {
        return Err(HmmerError::Format(format!(
            "{format_name} alignment contains no records"
        )));
    }
    let alen = aseq[0].len();
    for (name, seq) in sqname.iter().zip(&aseq) {
        if seq.len() != alen {
            return Err(HmmerError::Format(format!(
                "{format_name} sequence {name} has aligned length {}, expected {alen}",
                seq.len()
            )));
        }
    }

    let nseq = sqname.len();
    Ok(Msa {
        name: alignment_name,
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqdesc,
        weights: None,
        aseq,
        pp: vec![None; nseq],
        nseq,
        alen,
        rf: None,
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: None,
    })
}

fn flush_a2m_record(
    current_name: &mut Option<String>,
    current_desc: &mut String,
    current_raw: &mut Vec<u8>,
    records: &mut Vec<A2mRecord>,
) -> HmmerResult<()> {
    let Some(name) = current_name.take() else {
        return Ok(());
    };
    if current_raw.is_empty() {
        return Err(HmmerError::Format(format!(
            "A2M record {name} has no sequence data"
        )));
    }

    let mut inserts = vec![Vec::<u8>::new()];
    let mut consensus = Vec::<u8>::new();
    let mut current_insert = 0usize;
    for &raw in current_raw.iter() {
        if raw == b'.' {
            continue;
        }
        if raw.is_ascii_lowercase() {
            inserts[current_insert].push(raw);
        } else {
            consensus.push(raw);
            inserts.push(Vec::new());
            current_insert += 1;
        }
    }
    records.push(A2mRecord {
        name,
        desc: std::mem::take(current_desc),
        inserts,
        consensus,
    });
    current_raw.clear();
    Ok(())
}

fn flush_afa_record(
    current_name: &mut Option<String>,
    current_desc: &mut String,
    current_seq: &mut Vec<u8>,
    sqname: &mut Vec<String>,
    sqdesc: &mut Vec<String>,
    aseq: &mut Vec<Vec<u8>>,
) -> HmmerResult<()> {
    let Some(name) = current_name.take() else {
        return Ok(());
    };
    if current_seq.is_empty() {
        return Err(HmmerError::Format(format!(
            "aligned FASTA record {name} has no sequence data"
        )));
    }
    sqname.push(name);
    sqdesc.push(std::mem::take(current_desc));
    aseq.push(std::mem::take(current_seq));
    Ok(())
}

fn alignment_name_from_path(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != "-")
        .unwrap_or("alignment")
        .to_string()
}

/// Read all Stockholm-format alignments from an open reader.
///
/// Scans for `# STOCKHOLM` block headers and dispatches each block (terminated
/// by `//`) to `parse_stockholm_block`. Concatenated multi-MSA files are
/// supported. Lightweight Rust port of the Stockholm subset of
/// `esl_msafile_stockholm.c`.
pub fn read_stockholm_from_reader<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Msa>> {
    Ok(read_stockholm_preserved_from_reader(reader)?
        .into_iter()
        .map(|stockholm| stockholm.msa)
        .collect())
}

/// Read all Stockholm-format alignments from an open reader, preserving body
/// lines for writers that need to round-trip annotations.
pub fn read_stockholm_preserved_from_reader<R: Read>(
    reader: BufReader<R>,
) -> HmmerResult<Vec<StockholmMsa>> {
    let mut msas = Vec::new();
    let mut lines: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(HmmerError::Io)?;
        lines.push(line);
    }

    let mut i = 0;
    while i < lines.len() {
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
            if let Some(stockholm) = parse_stockholm_block(&lines[start..=end])? {
                msas.push(stockholm);
            }
        } else {
            let trimmed = lines[i].trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                return Err(HmmerError::Format(format!(
                    "Expected Stockholm header, got: {}",
                    trimmed
                )));
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
fn parse_stockholm_block(lines: &[String]) -> HmmerResult<Option<StockholmMsa>> {
    let mut name = String::new();
    let mut acc: Option<String> = None;
    let mut desc: Option<String> = None;
    let mut author: Option<String> = None;
    let mut cutoffs = StockholmCutoffs::default();
    let mut seq_order: Vec<String> = Vec::new();
    let mut seq_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut sqdesc: HashMap<String, String> = HashMap::new();
    let mut weights: HashMap<String, f64> = HashMap::new();
    let mut pp: HashMap<String, Vec<u8>> = HashMap::new();
    let mut rf: Option<Vec<u8>> = None;
    let mut mm: Option<Vec<u8>> = None;
    let mut ss_cons: Option<Vec<u8>> = None;
    let mut sa_cons: Option<Vec<u8>> = None;
    let mut pp_cons: Option<Vec<u8>> = None;
    let mut body_lines = Vec::new();
    let mut expected_block_order: Option<Vec<BlockLineKey>> = None;
    let mut current_block_order: Vec<BlockLineKey> = Vec::new();
    let mut current_block_names: HashSet<String> = HashSet::new();

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum BlockLineKey {
        Sequence(String),
        Gr { seq: String, tag: String },
        Gc(String),
    }

    impl BlockLineKey {
        fn label(&self) -> String {
            match self {
                Self::Sequence(name) => format!("sequence {name}"),
                Self::Gr { seq, tag } => format!("#=GR {seq} {tag}"),
                Self::Gc(tag) => format!("#=GC {tag}"),
            }
        }
    }

    fn finish_sequence_block(
        expected_block_order: &mut Option<Vec<BlockLineKey>>,
        current_block_order: &mut Vec<BlockLineKey>,
        current_block_names: &mut HashSet<String>,
    ) -> HmmerResult<()> {
        if current_block_order.is_empty() {
            return Ok(());
        }
        if let Some(expected) = expected_block_order.as_ref() {
            if current_block_order.len() != expected.len() {
                return Err(HmmerError::Format(format!(
                    "Stockholm sequence block has {} ordered lines, expected {}",
                    current_block_order.len(),
                    expected.len()
                )));
            }
        } else {
            *expected_block_order = Some(current_block_order.clone());
        }
        current_block_order.clear();
        current_block_names.clear();
        Ok(())
    }

    fn record_block_line(
        expected_block_order: &Option<Vec<BlockLineKey>>,
        current_block_order: &mut Vec<BlockLineKey>,
        key: BlockLineKey,
    ) -> HmmerResult<()> {
        if let Some(expected) = expected_block_order {
            let position = current_block_order.len();
            if expected.get(position) != Some(&key) {
                let expected_label = expected
                    .get(position)
                    .map(BlockLineKey::label)
                    .unwrap_or_else(|| "end of block".to_string());
                return Err(HmmerError::Format(format!(
                    "Stockholm block line {} is out of order in a later block: got {}, expected {}",
                    position + 1,
                    key.label(),
                    expected_label
                )));
            }
        }
        current_block_order.push(key);
        Ok(())
    }

    for line in lines {
        let trimmed = line.trim();
        if trimmed == "//" || trimmed.starts_with("# STOCKHOLM") {
            continue;
        }
        body_lines.push(line.clone());
        if trimmed.is_empty() {
            finish_sequence_block(
                &mut expected_block_order,
                &mut current_block_order,
                &mut current_block_names,
            )?;
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("#=GF ID") {
            let fields: Vec<&str> = rest.split_whitespace().collect();
            if fields.len() != 1 {
                return Err(HmmerError::Format(format!(
                    "Stockholm #=GF ID annotation must contain exactly one name token, got '{}'",
                    rest.trim()
                )));
            }
            name = fields[0].to_string();
        } else if let Some(rest) = trimmed.strip_prefix("#=GF AC") {
            acc = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("#=GF DE") {
            // C `stockholm_parse_gf` calls `esl_msa_SetDesc`, which frees the
            // existing value and overwrites it: the LAST `#=GF DE` line wins.
            // (esl_msa.c esl_msa_SetDesc.)
            desc = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("#=GF AU") {
            // Likewise C `esl_msa_SetAuthor` overwrites; the LAST `#=GF AU` wins.
            author = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("#=GF GA") {
            cutoffs.ga = Some(parse_stockholm_cutoffs("GA", rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("#=GF TC") {
            cutoffs.tc = Some(parse_stockholm_cutoffs("TC", rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("#=GF NC") {
            cutoffs.nc = Some(parse_stockholm_cutoffs("NC", rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("#=GC RF") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("RF".to_string()),
            )?;
            let rf_str = rest.trim();
            match &mut rf {
                Some(existing) => existing.extend_from_slice(rf_str.as_bytes()),
                None => rf = Some(rf_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC MM") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("MM".to_string()),
            )?;
            let mm_str = rest.trim();
            match &mut mm {
                Some(existing) => existing.extend_from_slice(mm_str.as_bytes()),
                None => mm = Some(mm_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC SS_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("SS_cons".to_string()),
            )?;
            let ss_str = rest.trim();
            match &mut ss_cons {
                Some(existing) => existing.extend_from_slice(ss_str.as_bytes()),
                None => ss_cons = Some(ss_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC SA_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("SA_cons".to_string()),
            )?;
            let sa_str = rest.trim();
            match &mut sa_cons {
                Some(existing) => existing.extend_from_slice(sa_str.as_bytes()),
                None => sa_cons = Some(sa_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC PP_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("PP_cons".to_string()),
            )?;
            let pp_str = rest.trim();
            match &mut pp_cons {
                Some(existing) => existing.extend_from_slice(pp_str.as_bytes()),
                None => pp_cons = Some(pp_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GS ") {
            let fields: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
            if fields.len() == 3 {
                if fields[1] == "DE" {
                    if sqdesc
                        .insert(fields[0].to_string(), fields[2].trim().to_string())
                        .is_some()
                    {
                        return Err(HmmerError::Format(format!(
                            "Stockholm sequence {} has more than one DE annotation",
                            fields[0]
                        )));
                    }
                } else if fields[1] == "WT" {
                    let weight = fields[2].trim().parse::<f64>().map_err(|e| {
                        HmmerError::Format(format!(
                            "Stockholm #=GS {} WT annotation is not a valid weight: {}",
                            fields[0], e
                        ))
                    })?;
                    if !weight.is_finite() {
                        return Err(HmmerError::Format(format!(
                            "Stockholm #=GS {} WT annotation must be a finite weight",
                            fields[0]
                        )));
                    }
                    if weights.insert(fields[0].to_string(), weight).is_some() {
                        return Err(HmmerError::Format(format!(
                            "Stockholm sequence {} has more than one WT annotation",
                            fields[0]
                        )));
                    }
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GR ") {
            let fields: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
            if fields.len() == 3 {
                record_block_line(
                    &expected_block_order,
                    &mut current_block_order,
                    BlockLineKey::Gr {
                        seq: fields[0].to_string(),
                        tag: fields[1].to_string(),
                    },
                )?;
            }
            if fields.len() == 3 && fields[1] == "PP" {
                pp.entry(fields[0].to_string())
                    .and_modify(|line| line.extend_from_slice(fields[2].trim().as_bytes()))
                    .or_insert_with(|| fields[2].trim().as_bytes().to_vec());
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC ") {
            if let Some(tag) = rest.split_whitespace().next() {
                record_block_line(
                    &expected_block_order,
                    &mut current_block_order,
                    BlockLineKey::Gc(tag.to_string()),
                )?;
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

                if !current_block_names.insert(sqname.clone()) {
                    return Err(HmmerError::Format(format!(
                        "Stockholm sequence {} occurs more than once in the same block",
                        sqname
                    )));
                }
                record_block_line(
                    &expected_block_order,
                    &mut current_block_order,
                    BlockLineKey::Sequence(sqname.clone()),
                )?;

                match seq_data.entry(sqname.clone()) {
                    Entry::Vacant(entry) => {
                        seq_order.push(sqname);
                        entry.insert(sqdata.to_vec());
                    }
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().extend_from_slice(sqdata);
                    }
                }
            }
        }
    }
    finish_sequence_block(
        &mut expected_block_order,
        &mut current_block_order,
        &mut current_block_names,
    )?;

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
    if let Some(ref mm) = mm {
        if mm.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC MM annotation has length {}, expected {}",
                mm.len(),
                alen
            )));
        }
    }
    if let Some(ref ss_cons) = ss_cons {
        if ss_cons.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC SS_cons annotation has length {}, expected {}",
                ss_cons.len(),
                alen
            )));
        }
    }
    if let Some(ref sa_cons) = sa_cons {
        if sa_cons.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC SA_cons annotation has length {}, expected {}",
                sa_cons.len(),
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
    for name in weights.keys() {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GS {} WT annotation refers to unknown sequence",
                name
            )));
        }
    }
    if !weights.is_empty() && weights.len() != seq_order.len() {
        if let Some(name) = seq_order.iter().find(|name| !weights.contains_key(*name)) {
            return Err(HmmerError::Format(format!(
                "Stockholm record ended without a weight for {}",
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
    let weight_vec = if weights.is_empty() {
        None
    } else {
        Some(
            seq_order
                .iter()
                .map(|name| weights.remove(name).unwrap())
                .collect(),
        )
    };
    let pp_vec: Vec<Option<Vec<u8>>> = seq_order.iter().map(|name| pp.remove(name)).collect();

    Ok(Some(StockholmMsa {
        msa: Msa {
            name,
            acc,
            desc,
            author,
            sqname: seq_order,
            sqdesc: sqdesc_vec,
            weights: weight_vec,
            pp: pp_vec,
            aseq,
            nseq,
            alen,
            rf,
            mm,
            ss_cons,
            sa_cons,
            pp_cons,
        },
        cutoffs,
        body_lines,
    }))
}

fn parse_stockholm_cutoffs(tag: &str, value: &str) -> HmmerResult<[f32; 2]> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let Some(first) = parts.first() else {
        return Err(HmmerError::Format(format!(
            "Missing Stockholm #=GF {tag} cutoff value"
        )));
    };
    let first = parse_stockholm_cutoff_value(tag, first)?;
    let second = match parts.get(1) {
        Some(value) => parse_stockholm_cutoff_value(tag, value)?,
        None => first,
    };
    Ok([first, second])
}

fn parse_stockholm_cutoff_value(tag: &str, value: &str) -> HmmerResult<f32> {
    value
        .trim_end_matches(';')
        .parse()
        .map_err(|_| HmmerError::Format(format!("Bad Stockholm #=GF {tag} cutoff value: {value}")))
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

/// A full-fidelity Stockholm annotation model, mirroring the subset of Easel's
/// `ESL_MSA` that the text-mode Stockholm writer (`stockholm_write`) consumes.
///
/// This is a parallel, additive model to [`Msa`]/[`StockholmMsa`]: it captures
/// every annotation type and ordering needed to re-serialize a Stockholm file
/// byte-for-byte the way C does (used by `alimask`). It is read in *text mode*:
/// residue / GR / GC characters pass through verbatim (no gap-character
/// normalization), matching `alimask`, which opens the alignment with a NULL
/// alphabet.
#[derive(Debug, Clone, Default)]
pub struct FullStockholm {
    /// Free-text comment lines (`#<comment>`), with the leading `#` and
    /// following whitespace stripped (as in `stockholm_parse_comment`).
    pub comments: Vec<String>,
    /// `#=GF ID` alignment name.
    pub name: Option<String>,
    /// `#=GF AC` accession.
    pub acc: Option<String>,
    /// `#=GF DE` description (last line wins, like `esl_msa_SetDesc`).
    pub desc: Option<String>,
    /// `#=GF AU` author (last line wins, like `esl_msa_SetAuthor`).
    pub au: Option<String>,
    /// `#=GF GA` gathering cutoffs (one or two values).
    pub ga: Option<(f32, Option<f32>)>,
    /// `#=GF NC` noise cutoffs.
    pub nc: Option<(f32, Option<f32>)>,
    /// `#=GF TC` trusted cutoffs.
    pub tc: Option<(f32, Option<f32>)>,
    /// Remaining `#=GF` tags, in input order: `(tag, value)`.
    pub gf: Vec<(String, String)>,

    /// Sequence names, in first-seen order.
    pub sqname: Vec<String>,
    /// Aligned sequence rows (text, verbatim), parallel to `sqname`.
    pub aseq: Vec<Vec<u8>>,
    /// Whether any `#=GS <seq> WT` weight was seen (`eslMSA_HASWGTS`).
    pub has_wgts: bool,
    /// Per-sequence weights (parallel to `sqname`); valid only if `has_wgts`.
    pub wgt: Vec<f64>,
    /// Per-sequence `#=GS <seq> AC` accession (None if unset).
    pub sqacc: Vec<Option<String>>,
    /// Per-sequence `#=GS <seq> DE` description (None if unset).
    pub sqdesc: Vec<Option<String>>,
    /// Remaining `#=GS` tags, in first-seen order. Each entry is
    /// `(tag, per-seq values)` where the per-seq vector is parallel to
    /// `sqname`; each value is `\n`-joined when a sequence has multiple lines
    /// of the same tag (e.g. `DR PDB;`), matching `esl_msa_AddGS`.
    pub gs: Vec<(String, Vec<Option<String>>)>,

    /// Per-sequence `#=GR <seq> SS` (parallel to `sqname`).
    pub ss: Vec<Option<Vec<u8>>>,
    /// Per-sequence `#=GR <seq> SA`.
    pub sa: Vec<Option<Vec<u8>>>,
    /// Per-sequence `#=GR <seq> PP`.
    pub pp: Vec<Option<Vec<u8>>>,
    /// Remaining `#=GR` tags, first-seen order: `(tag, per-seq values)`.
    pub gr: Vec<(String, Vec<Option<Vec<u8>>>)>,

    /// `#=GC SS_cons`.
    pub ss_cons: Option<Vec<u8>>,
    /// `#=GC SA_cons`.
    pub sa_cons: Option<Vec<u8>>,
    /// `#=GC PP_cons`.
    pub pp_cons: Option<Vec<u8>>,
    /// `#=GC RF`.
    pub rf: Option<Vec<u8>>,
    /// `#=GC MM`.
    pub mm: Option<Vec<u8>>,
    /// Remaining `#=GC` tags, first-seen order: `(tag, value)`.
    pub gc: Vec<(String, Vec<u8>)>,

    /// Alignment length (columns).
    pub alen: usize,
}

impl FullStockholm {
    fn seqidx(&mut self, name: &str) -> usize {
        if let Some(i) = self.sqname.iter().position(|n| n == name) {
            return i;
        }
        self.sqname.push(name.to_string());
        self.aseq.push(Vec::new());
        self.wgt.push(-1.0);
        self.sqacc.push(None);
        self.sqdesc.push(None);
        self.ss.push(None);
        self.sa.push(None);
        self.pp.push(None);
        for (_, vals) in &mut self.gs {
            vals.push(None);
        }
        for (_, vals) in &mut self.gr {
            vals.push(None);
        }
        self.sqname.len() - 1
    }

    fn gs_tagidx(&mut self, tag: &str) -> usize {
        if let Some(i) = self.gs.iter().position(|(t, _)| t == tag) {
            return i;
        }
        self.gs
            .push((tag.to_string(), vec![None; self.sqname.len()]));
        self.gs.len() - 1
    }

    fn gr_tagidx(&mut self, tag: &str) -> usize {
        if let Some(i) = self.gr.iter().position(|(t, _)| t == tag) {
            return i;
        }
        self.gr
            .push((tag.to_string(), vec![None; self.sqname.len()]));
        self.gr.len() - 1
    }

    fn gc_tagidx(&mut self, tag: &str) -> usize {
        if let Some(i) = self.gc.iter().position(|(t, _)| t == tag) {
            return i;
        }
        self.gc.push((tag.to_string(), Vec::new()));
        self.gc.len() - 1
    }
}

/// Read every Stockholm alignment in `path` into the full-fidelity
/// [`FullStockholm`] model (text mode).
pub fn read_stockholm_full(path: &Path) -> HmmerResult<Vec<FullStockholm>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    read_stockholm_full_from_reader(BufReader::new(file))
}

/// Read every Stockholm alignment from a reader into [`FullStockholm`] models.
///
/// Faithful text-mode port of the bucketing performed by the Easel Stockholm
/// reader (`stockholm_parse_{gf,gs,gc,gr,sq,comment}`). One `FullStockholm` is
/// returned per `# STOCKHOLM`/`//` record.
pub fn read_stockholm_full_from_reader<R: Read>(
    reader: BufReader<R>,
) -> HmmerResult<Vec<FullStockholm>> {
    let mut out = Vec::new();
    let mut cur: Option<FullStockholm> = None;

    for line in reader.lines() {
        let line = line.map_err(HmmerError::Io)?;
        // Strip leading whitespace, as the C reader does before dispatch.
        let p = line.trim_start_matches([' ', '\t']);

        if cur.is_none() {
            // Skip leading blank/comment lines until a Stockholm header.
            if p.is_empty() {
                continue;
            }
            if p.starts_with("# STOCKHOLM") {
                cur = Some(FullStockholm::default());
                continue;
            }
            if p.starts_with('#') {
                continue;
            }
            return Err(HmmerError::Format(format!(
                "Expected Stockholm header, got: {p}"
            )));
        }

        if p == "//" {
            let mut msa = cur.take().unwrap();
            finalize_full(&mut msa)?;
            out.push(msa);
            continue;
        }
        if p.starts_with("# STOCKHOLM") {
            continue;
        }
        let msa = cur.as_mut().unwrap();
        if p.is_empty() {
            continue;
        }
        if let Some(rest) = p.strip_prefix('#') {
            if let Some(body) = rest.strip_prefix("=GF") {
                parse_full_gf(msa, body)?;
            } else if let Some(body) = rest.strip_prefix("=GS") {
                parse_full_gs(msa, body)?;
            } else if let Some(body) = rest.strip_prefix("=GC") {
                parse_full_gc(msa, body)?;
            } else if let Some(body) = rest.strip_prefix("=GR") {
                parse_full_gr(msa, body)?;
            } else {
                // Free-text comment: strip leading whitespace after '#'.
                msa.comments
                    .push(rest.trim_start_matches([' ', '\t']).to_string());
            }
        } else {
            // Sequence line: name aseq
            let mut it = p.splitn(2, [' ', '\t']);
            let name = it.next().unwrap_or("");
            let seq = it.next().unwrap_or("").trim_start_matches([' ', '\t']);
            if name.is_empty() {
                continue;
            }
            let idx = msa.seqidx(name);
            msa.aseq[idx].extend_from_slice(seq.as_bytes());
        }
    }

    if cur.is_some() {
        return Err(HmmerError::Format(
            "missing // terminator after MSA".to_string(),
        ));
    }
    Ok(out)
}

/// Tokenize Stockholm markup into (first whitespace-delimited token, remainder
/// with leading whitespace stripped, trailing whitespace preserved), matching
/// `esl_memtok` (which skips the trailing delimiter run after the token).
fn split_tok(s: &str) -> (&str, &str) {
    let s = s.trim_start_matches([' ', '\t']);
    match s.find([' ', '\t']) {
        Some(i) => (&s[..i], s[i..].trim_start_matches([' ', '\t'])),
        None => (s, ""),
    }
}

fn parse_cutoffs(value: &str) -> HmmerResult<(f32, Option<f32>)> {
    // C `esl_memtof` parses a leading real number and ignores trailing junk
    // such as the `;` Rfam/Pfam append (e.g. "27.00 27.00;").
    fn parse_real(tok: &str) -> HmmerResult<f32> {
        let trimmed = tok.trim_end_matches(';');
        trimmed
            .parse::<f32>()
            .map_err(|_| HmmerError::Format(format!("bad #=GF cutoff value: {tok}")))
    }
    let mut it = value.split([' ', '\t']).filter(|t| !t.is_empty());
    let first = it
        .next()
        .ok_or_else(|| HmmerError::Format("missing #=GF cutoff value".to_string()))?;
    let first = parse_real(first)?;
    let second = match it.next() {
        Some(v) => Some(parse_real(v)?),
        None => None,
    };
    Ok((first, second))
}

fn parse_full_gf(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (tag, value) = split_tok(body);
    match tag {
        "ID" => msa.name = Some(split_tok(value).0.to_string()),
        "AC" => msa.acc = Some(split_tok(value).0.to_string()),
        "DE" => msa.desc = Some(value.to_string()),
        "AU" => msa.au = Some(value.to_string()),
        "GA" => msa.ga = Some(parse_cutoffs(value)?),
        "NC" => msa.nc = Some(parse_cutoffs(value)?),
        "TC" => msa.tc = Some(parse_cutoffs(value)?),
        _ => msa.gf.push((tag.to_string(), value.to_string())),
    }
    Ok(())
}

fn parse_full_gs(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (seqname, rest) = split_tok(body);
    let (tag, value) = split_tok(rest);
    if seqname.is_empty() || tag.is_empty() {
        return Ok(());
    }
    let idx = msa.seqidx(seqname);
    match tag {
        "WT" => {
            let w: f64 = split_tok(value)
                .0
                .parse()
                .map_err(|_| HmmerError::Format("bad #=GS WT value".to_string()))?;
            msa.wgt[idx] = w;
            msa.has_wgts = true;
        }
        "AC" => msa.sqacc[idx] = Some(split_tok(value).0.to_string()),
        "DE" => msa.sqdesc[idx] = Some(value.to_string()),
        _ => {
            let t = msa.gs_tagidx(tag);
            let slot = &mut msa.gs[t].1[idx];
            match slot {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(value);
                }
                None => *slot = Some(value.to_string()),
            }
        }
    }
    Ok(())
}

fn parse_full_gc(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (tag, value) = split_tok(body);
    if tag.is_empty() {
        return Ok(());
    }
    let bytes = value.as_bytes();
    let append = |slot: &mut Option<Vec<u8>>| match slot {
        Some(v) => v.extend_from_slice(bytes),
        None => *slot = Some(bytes.to_vec()),
    };
    match tag {
        "SS_cons" => append(&mut msa.ss_cons),
        "SA_cons" => append(&mut msa.sa_cons),
        "PP_cons" => append(&mut msa.pp_cons),
        "RF" => append(&mut msa.rf),
        "MM" => append(&mut msa.mm),
        _ => {
            let t = msa.gc_tagidx(tag);
            msa.gc[t].1.extend_from_slice(bytes);
        }
    }
    Ok(())
}

fn parse_full_gr(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (seqname, rest) = split_tok(body);
    let (tag, value) = split_tok(rest);
    if seqname.is_empty() || tag.is_empty() {
        return Ok(());
    }
    let idx = msa.seqidx(seqname);
    let bytes = value.as_bytes();
    let append = |slot: &mut Option<Vec<u8>>| match slot {
        Some(v) => v.extend_from_slice(bytes),
        None => *slot = Some(bytes.to_vec()),
    };
    match tag {
        "SS" => append(&mut msa.ss[idx]),
        "SA" => append(&mut msa.sa[idx]),
        "PP" => append(&mut msa.pp[idx]),
        _ => {
            let t = msa.gr_tagidx(tag);
            append(&mut msa.gr[t].1[idx]);
        }
    }
    Ok(())
}

fn finalize_full(msa: &mut FullStockholm) -> HmmerResult<()> {
    msa.alen = msa.aseq.first().map(|s| s.len()).unwrap_or(0);
    Ok(())
}

/// Maximum byte-length of a set of strings (Easel `esl_str_GetMaxWidth`).
fn max_width<I: IntoIterator<Item = usize>>(lens: I) -> usize {
    lens.into_iter().max().unwrap_or(0)
}

/// Serialize a [`FullStockholm`] to `out`, byte-for-byte matching Easel's
/// `stockholm_write(fp, msa, cpl)` (`esl_msafile_stockholm.c`).
///
/// `cpl` is the characters-per-line block width (200 for `eslMSAFILE_STOCKHOLM`,
/// as `esl_msafile_Write` passes for `alimask`).
///
/// If `abc` is `Some`, the aligned sequences are textized through it (the
/// digital-mode path, line 1231: `esl_abc_TextizeN`): each residue is mapped to
/// its canonical symbol and every gap character (`.`, `_`, `-`) normalizes to
/// `-`. This matches `alimask`, which opens the alignment with an autodetected
/// alphabet (digital mode). If `abc` is `None`, sequences are written verbatim
/// (text mode, line 1232). Annotation lines (`#=GR`/`#=GC`) are always text and
/// never normalized, exactly as in C.
pub fn write_stockholm_full(
    out: &mut dyn std::io::Write,
    msa: &FullStockholm,
    cpl: usize,
    abc: Option<&crate::alphabet::Alphabet>,
) -> std::io::Result<()> {
    // In digital mode, sequences are textized through the alphabet: residues
    // map to canonical symbols and all gap chars normalize to '-'. Annotation
    // (#=GR/#=GC) stays text in either mode, so we only transform aseq.
    let textized: Vec<Vec<u8>> = match abc {
        Some(abc) => msa
            .aseq
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&ch| {
                        if ch == b'-' || ch == b'.' || ch == b'_' {
                            abc.sym[abc.gap_code() as usize]
                        } else {
                            let code = abc.digitize_symbol(ch);
                            if code == crate::alphabet::DSQ_ILLEGAL
                                || code == crate::alphabet::DSQ_IGNORED
                            {
                                ch
                            } else {
                                abc.sym[code as usize]
                            }
                        }
                    })
                    .collect()
            })
            .collect(),
        None => msa.aseq.clone(),
    };

    // Unique-name check (esl_msa_CheckUniqueNames). If names collide, C forces
    // a "<i>|" prefix of width uniqwidth.
    let mut uniqwidth = 0usize;
    let make_unique = {
        let mut seen = HashSet::new();
        let dup = msa.sqname.iter().any(|n| !seen.insert(n.as_str()));
        if dup {
            let mut t = msa.sqname.len();
            while t != 0 {
                uniqwidth += 1;
                t /= 10;
            }
            uniqwidth += 1; // includes the '|'
        }
        dup
    };

    let maxname = max_width(msa.sqname.iter().map(|n| n.len()));

    let mut maxgf = max_width(msa.gf.iter().map(|(t, _)| t.len()));
    if maxgf < 2 {
        maxgf = 2;
    }

    let mut maxgc = max_width(msa.gc.iter().map(|(t, _)| t.len()));
    if (msa.rf.is_some() || msa.mm.is_some()) && maxgc < 2 {
        maxgc = 2;
    }
    if (msa.ss_cons.is_some() || msa.sa_cons.is_some() || msa.pp_cons.is_some()) && maxgc < 7 {
        maxgc = 7;
    }

    let mut maxgr = max_width(msa.gr.iter().map(|(t, _)| t.len()));
    let any_ss = msa.ss.iter().any(Option::is_some);
    let any_sa = msa.sa.iter().any(Option::is_some);
    let any_pp = msa.pp.iter().any(Option::is_some);
    if (any_ss || any_sa || any_pp) && maxgr < 2 {
        maxgr = 2;
    }

    let mut margin = uniqwidth + maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && uniqwidth + maxname + maxgr + 7 > margin {
        margin = uniqwidth + maxname + maxgr + 7;
    }

    // Helper: write a left-justified seqname (with optional uniqizing prefix).
    let write_name =
        |out: &mut dyn std::io::Write, i: usize, field: usize| -> std::io::Result<()> {
            if make_unique {
                // "%0*d|%-*s" with uniqwidth-1 and field
                write!(
                    out,
                    "{:0width$}|{:<field$}",
                    i,
                    msa.sqname[i],
                    width = uniqwidth - 1,
                    field = field
                )
            } else {
                write!(out, "{:<field$}", msa.sqname[i], field = field)
            }
        };

    writeln!(out, "# STOCKHOLM 1.0")?;
    if make_unique {
        writeln!(
            out,
            "# WARNING: seq names have been made unique by adding a prefix of \"<seq#>|\""
        )?;
    }

    // Comments
    for c in &msa.comments {
        writeln!(out, "#{c}")?;
    }
    if !msa.comments.is_empty() {
        writeln!(out)?;
    }

    // GF section
    if let Some(name) = &msa.name {
        writeln!(out, "#=GF {:<maxgf$} {}", "ID", name)?;
    }
    if let Some(acc) = &msa.acc {
        writeln!(out, "#=GF {:<maxgf$} {}", "AC", acc)?;
    }
    if let Some(desc) = &msa.desc {
        writeln!(out, "#=GF {:<maxgf$} {}", "DE", desc)?;
    }
    if let Some(au) = &msa.au {
        writeln!(out, "#=GF {:<maxgf$} {}", "AU", au)?;
    }
    write_cutoff(out, "GA", msa.ga, maxgf)?;
    write_cutoff(out, "NC", msa.nc, maxgf)?;
    write_cutoff(out, "TC", msa.tc, maxgf)?;
    for (tag, value) in &msa.gf {
        writeln!(out, "#=GF {:<maxgf$} {}", tag, value)?;
    }
    writeln!(out)?;

    // GS section
    if msa.has_wgts {
        for i in 0..msa.sqname.len() {
            write!(out, "#=GS ")?;
            write_name(out, i, maxname)?;
            writeln!(out, " WT {:.2}", msa.wgt[i])?;
        }
        writeln!(out)?;
    }
    if msa.sqacc.iter().any(Option::is_some) {
        for i in 0..msa.sqname.len() {
            if let Some(acc) = &msa.sqacc[i] {
                write!(out, "#=GS ")?;
                write_name(out, i, maxname)?;
                writeln!(out, " AC {}", acc)?;
            }
        }
        writeln!(out)?;
    }
    if msa.sqdesc.iter().any(Option::is_some) {
        for i in 0..msa.sqname.len() {
            if let Some(de) = &msa.sqdesc[i] {
                write!(out, "#=GS ")?;
                write_name(out, i, maxname)?;
                writeln!(out, " DE {}", de)?;
            }
        }
        writeln!(out)?;
    }
    for (tag, vals) in &msa.gs {
        let gslen = tag.len();
        for (i, v) in vals.iter().enumerate() {
            if let Some(v) = v {
                for tok in v.split('\n') {
                    write!(out, "#=GS ")?;
                    write_name(out, i, maxname)?;
                    writeln!(out, " {:<gslen$} {}", tag, tok)?;
                }
            }
        }
        writeln!(out)?;
    }

    // Alignment blocks
    let alen = msa.alen;
    let mut currpos = 0usize;
    let mut first_block = true;
    while currpos < alen {
        let acpl = (alen - currpos).min(cpl);
        let end = currpos + acpl;
        if !first_block {
            writeln!(out)?;
        }
        first_block = false;

        for i in 0..msa.sqname.len() {
            write_name(out, i, margin - uniqwidth - 1)?;
            out.write_all(b" ")?;
            out.write_all(&textized[i][currpos..end])?;
            writeln!(out)?;

            write_gr(
                out,
                msa,
                i,
                "SS",
                &msa.ss[i],
                maxname,
                margin,
                uniqwidth,
                currpos,
                end,
                make_unique,
            )?;
            write_gr(
                out,
                msa,
                i,
                "SA",
                &msa.sa[i],
                maxname,
                margin,
                uniqwidth,
                currpos,
                end,
                make_unique,
            )?;
            write_gr(
                out,
                msa,
                i,
                "PP",
                &msa.pp[i],
                maxname,
                margin,
                uniqwidth,
                currpos,
                end,
                make_unique,
            )?;
            for (tag, vals) in &msa.gr {
                write_gr(
                    out,
                    msa,
                    i,
                    tag,
                    &vals[i],
                    maxname,
                    margin,
                    uniqwidth,
                    currpos,
                    end,
                    make_unique,
                )?;
            }
        }

        write_gc(out, "SS_cons", &msa.ss_cons, margin, currpos, end)?;
        write_gc(out, "SA_cons", &msa.sa_cons, margin, currpos, end)?;
        write_gc(out, "PP_cons", &msa.pp_cons, margin, currpos, end)?;
        write_gc(out, "RF", &msa.rf, margin, currpos, end)?;
        write_gc(out, "MM", &msa.mm, margin, currpos, end)?;
        for (tag, value) in &msa.gc {
            write_gc_bytes(out, tag, value, margin, currpos, end)?;
        }

        currpos += cpl;
    }
    writeln!(out, "//")?;
    Ok(())
}

fn write_cutoff(
    out: &mut dyn std::io::Write,
    tag: &str,
    cut: Option<(f32, Option<f32>)>,
    maxgf: usize,
) -> std::io::Result<()> {
    if let Some((a, b)) = cut {
        match b {
            Some(b) => writeln!(out, "#=GF {:<maxgf$} {:.1} {:.1}", tag, a, b),
            None => writeln!(out, "#=GF {:<maxgf$} {:.1}", tag, a),
        }
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn write_gr(
    out: &mut dyn std::io::Write,
    msa: &FullStockholm,
    i: usize,
    tag: &str,
    data: &Option<Vec<u8>>,
    maxname: usize,
    margin: usize,
    uniqwidth: usize,
    currpos: usize,
    end: usize,
    make_unique: bool,
) -> std::io::Result<()> {
    let Some(data) = data else { return Ok(()) };
    let tagfield = margin - maxname - uniqwidth - 7;
    write!(out, "#=GR ")?;
    if make_unique {
        write!(
            out,
            "{:0width$}|{:<field$}",
            i,
            msa.sqname[i],
            width = uniqwidth - 1,
            field = maxname
        )?;
    } else {
        write!(out, "{:<maxname$}", msa.sqname[i])?;
    }
    write!(out, " {:<tagfield$} ", tag)?;
    out.write_all(&data[currpos..end])?;
    writeln!(out)
}

fn write_gc(
    out: &mut dyn std::io::Write,
    tag: &str,
    data: &Option<Vec<u8>>,
    margin: usize,
    currpos: usize,
    end: usize,
) -> std::io::Result<()> {
    match data {
        Some(d) => write_gc_bytes(out, tag, d, margin, currpos, end),
        None => Ok(()),
    }
}

fn write_gc_bytes(
    out: &mut dyn std::io::Write,
    tag: &str,
    data: &[u8],
    margin: usize,
    currpos: usize,
    end: usize,
) -> std::io::Result<()> {
    let field = margin - 6;
    write!(out, "#=GC {:<field$} ", tag)?;
    out.write_all(&data[currpos..end])?;
    writeln!(out)
}

/// A single aligned row to be serialized by the text-mode MSA writers below.
///
/// These writers operate on already-assembled text alignment rows (as produced
/// by `hmmalign`), independent of the [`Msa`] reader struct. `aseq` holds the
/// aligned text (consensus columns + insert columns, with gap characters).
pub struct WriteRow<'a> {
    pub name: &'a str,
    pub acc: Option<&'a str>,
    pub desc: Option<&'a str>,
    pub aseq: &'a str,
}

/// Write an aligned FASTA (AFA) alignment to `out`.
///
/// Faithful port of Easel's text-mode `esl_msafile_afa_Write`: each record is a
/// `>name [acc] [desc]` header followed by the aligned sequence wrapped at 60
/// columns per line. Aligned text (including gap characters) is emitted verbatim.
pub fn write_afa(out: &mut dyn std::io::Write, rows: &[WriteRow<'_>]) -> std::io::Result<()> {
    for row in rows {
        write!(out, ">{}", row.name)?;
        if let Some(acc) = row.acc {
            if !acc.is_empty() {
                write!(out, " {}", acc)?;
            }
        }
        if let Some(desc) = row.desc {
            if !desc.is_empty() {
                write!(out, " {}", desc)?;
            }
        }
        writeln!(out)?;
        for chunk in row.aseq.as_bytes().chunks(60) {
            out.write_all(chunk)?;
            writeln!(out)?;
        }
    }
    Ok(())
}

/// Write a CLUSTAL (or CLUSTAL-like) alignment to `out`.
///
/// Faithful port of Easel's text-mode `esl_msafile_clustal_Write`. Emits the
/// magic header (`CLUSTAL 2.1 ...` or `EASEL (...) ...`), then interleaved
/// blocks of 60 columns. Each block is preceded by a blank line, and each block
/// ends with a consensus line: in text mode (where the alphabet is unknown),
/// only `*` (a column with exactly one distinct alphabetic symbol) and ` `
/// appear, matching `make_text_consensus_line`.
pub fn write_clustal(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    clustallike: bool,
    easel_version: &str,
) -> std::io::Result<()> {
    let cpl = 60usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);
    let maxnamelen = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    let consline = text_consensus_line(rows, alen);

    if clustallike {
        writeln!(out, "EASEL ({}) multiple sequence alignment", easel_version)?;
    } else {
        writeln!(out, "CLUSTAL 2.1 multiple sequence alignment")?;
    }

    let mut apos = 0usize;
    while apos < alen {
        let end = (apos + cpl).min(alen);
        writeln!(out)?;
        for row in rows {
            let seg = &row.aseq.as_bytes()[apos..end];
            write!(out, "{:<width$} ", row.name, width = maxnamelen)?;
            out.write_all(seg)?;
            writeln!(out)?;
        }
        write!(out, "{:<width$} ", "", width = maxnamelen)?;
        out.write_all(&consline[apos..end])?;
        writeln!(out)?;
        apos += cpl;
    }
    Ok(())
}

/// Build the text-mode CLUSTAL consensus line (`esl_msafile_clustal.c`
/// `make_text_consensus_line`). A column gets `*` iff exactly one distinct
/// uppercased A-Z symbol appears in it (no gaps or non-alphabetic symbols);
/// otherwise ` `.
fn text_consensus_line(rows: &[WriteRow<'_>], alen: usize) -> Vec<u8> {
    let mut v = vec![0u32; alen];
    for row in rows {
        for (apos, &ch) in row.aseq.as_bytes().iter().enumerate().take(alen) {
            let x = ch.to_ascii_uppercase() as i32 - b'A' as i32;
            if (0..26).contains(&x) {
                v[apos] |= 1 << x;
            } else {
                v[apos] |= 1 << 26;
            }
        }
    }
    let maxv = (1u32 << 26) - 1;
    let mut consline = vec![b' '; alen];
    for apos in 0..alen {
        let nbits = v[apos].count_ones();
        consline[apos] = if nbits == 1 && v[apos] < maxv {
            b'*'
        } else {
            b' '
        };
    }
    consline
}

/// Write a PSIBLAST alignment to `out`.
///
/// Faithful port of the text-mode `esl_msafile_psiblast_Write`: interleaved
/// blocks of 60 columns, `name  aseq` rows (names left-justified to a common
/// width followed by two spaces), blocks separated by a blank line.
///
/// Each column is classified as consensus or insert from `rf` (a column is
/// consensus iff `rf[col]` is alphanumeric; if `rf` is `None`, the first row is
/// used as the reference, matching Easel). Within each column an alphanumeric
/// symbol is uppercased in consensus columns and lowercased in insert columns;
/// every non-residue character becomes `-`.
pub fn write_psiblast(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    rf: Option<&str>,
) -> std::io::Result<()> {
    let cpl = 60usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);
    let maxnamelen = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    let rf_bytes = rf.map(|s| s.as_bytes());
    let first = rows.first().map(|r| r.aseq.as_bytes());

    let is_consensus = |col: usize| -> bool {
        match rf_bytes {
            Some(rf) => rf.get(col).is_some_and(|c| c.is_ascii_alphanumeric()),
            None => first.is_some_and(|f| f.get(col).is_some_and(|c| c.is_ascii_alphanumeric())),
        }
    };

    let mut pos = 0usize;
    while pos < alen {
        let end = (pos + cpl).min(alen);
        for row in rows {
            let seq = row.aseq.as_bytes();
            let mut buf = Vec::with_capacity(end - pos);
            for (offset, &sym) in seq[pos..end].iter().enumerate() {
                let col = pos + offset;
                let is_residue = sym.is_ascii_alphanumeric();
                let ch = if is_consensus(col) {
                    if is_residue {
                        sym.to_ascii_uppercase()
                    } else {
                        b'-'
                    }
                } else if is_residue {
                    sym.to_ascii_lowercase()
                } else {
                    b'-'
                };
                buf.push(ch);
            }
            write!(out, "{:<width$}  ", row.name, width = maxnamelen)?;
            out.write_all(&buf)?;
            writeln!(out)?;
        }
        if end < alen {
            writeln!(out)?;
        }
        pos += cpl;
    }
    Ok(())
}

/// Write a SELEX alignment to `out`.
///
/// Faithful port of the text-mode core of `esl_msafile_selex_Write`: interleaved
/// blocks of 60 columns, each row written as `name aseq` with names
/// left-justified to a common width (minimum 4, to accommodate `#=RF` etc.).
/// Blocks after the first are separated by a blank line. An optional `rf`
/// reference line is written as `#=RF` at the top of each block.
pub fn write_selex(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    rf: Option<&str>,
) -> std::io::Result<()> {
    let cpl = 60usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);
    let maxnamelen = rows.iter().map(|r| r.name.len()).max().unwrap_or(0).max(4);

    let mut apos = 0usize;
    while apos < alen {
        let end = (apos + cpl).min(alen);
        if apos != 0 {
            writeln!(out)?;
        }
        if let Some(rf) = rf {
            write!(out, "{:<width$} ", "#=RF", width = maxnamelen)?;
            out.write_all(&rf.as_bytes()[apos..end.min(rf.len())])?;
            writeln!(out)?;
        }
        for row in rows {
            write!(out, "{:<width$} ", row.name, width = maxnamelen)?;
            out.write_all(&row.aseq.as_bytes()[apos..end])?;
            writeln!(out)?;
        }
        apos += cpl;
    }
    Ok(())
}

/// Write a PHYLIP alignment to `out` (interleaved if `sequential` is false,
/// sequential otherwise).
///
/// Faithful port of the text-mode `phylip_interleaved_Write` /
/// `phylip_sequential_Write`. Header is ` <nseq> <alen>`. Names are truncated /
/// padded to `namewidth` (strict PHYLIP = 10) and only appear on the first line
/// of each sequence. Residues per line default to 60. Text symbols are rectified
/// per `phylip_rectify_output_seq_text`: lowercase to uppercase, `.`/`_`/space to
/// `-`, and `~` to `?`.
pub fn write_phylip(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    sequential: bool,
) -> std::io::Result<()> {
    let rpl = 60usize;
    let namewidth = 10usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);

    let rectify = |seg: &[u8]| -> Vec<u8> {
        seg.iter()
            .map(|&ch| {
                let ch = ch.to_ascii_uppercase();
                match ch {
                    b'.' | b'_' | b' ' => b'-',
                    b'~' => b'?',
                    other => other,
                }
            })
            .collect()
    };
    let padded_name = |name: &str| -> Vec<u8> {
        let mut bytes = name.as_bytes().to_vec();
        bytes.truncate(namewidth);
        while bytes.len() < namewidth {
            bytes.push(b' ');
        }
        bytes
    };

    if sequential {
        writeln!(out, " {} {}", rows.len(), alen)?;
        for row in rows {
            let seq = row.aseq.as_bytes();
            let mut apos = 0usize;
            while apos < alen {
                let end = (apos + rpl).min(alen);
                let seg = rectify(&seq[apos..end]);
                if apos == 0 {
                    out.write_all(&padded_name(row.name))?;
                    write!(out, " ")?;
                }
                out.write_all(&seg)?;
                writeln!(out)?;
                apos += rpl;
            }
        }
    } else {
        write!(out, " {} {}", rows.len(), alen)?;
        let mut apos = 0usize;
        while apos < alen {
            let end = (apos + rpl).min(alen);
            writeln!(out)?;
            for row in rows {
                let seg = rectify(&row.aseq.as_bytes()[apos..end]);
                if apos == 0 {
                    out.write_all(&padded_name(row.name))?;
                    write!(out, " ")?;
                }
                out.write_all(&seg)?;
                writeln!(out)?;
            }
            apos += rpl;
        }
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
    fn stockholm_de_and_au_last_line_wins() {
        // C `esl_msa_SetDesc`/`SetAuthor` (called from stockholm_parse_gf) free
        // and overwrite the field: the LAST #=GF DE / #=GF AU line wins. Verified
        // with `esl-reformat stockholm` on this input (emits "second part" /
        // "Author Two" only).
        let input = b"# STOCKHOLM 1.0\n#=GF ID testaln\n#=GF DE first part\n#=GF DE second part\n#=GF AU Author One\n#=GF AU Author Two\nseq1 ACDEF\nseq2 ACDEG\n//\n";
        let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
        let msa = &msas[0];
        assert_eq!(msa.desc.as_deref(), Some("second part"));
        assert_eq!(msa.author.as_deref(), Some("Author Two"));
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
        assert!(msa.weights.is_none());
        assert_eq!(msa.pp[0].as_deref(), Some(&b"9*"[..]));
        assert_eq!(msa.pp_cons.as_deref(), Some(&b"8*"[..]));
    }

    #[test]
    fn parses_stockholm_author_and_cutoffs() {
        let input = b"# STOCKHOLM 1.0\n#=GF AU first author\n#=GF AU second author\n#=GF GA 25.0 24.5\n#=GF TC 30.0;\n#=GF NC -1.0 -2.0;\ns1 AC\n//\n";
        let records = read_stockholm_preserved_from_reader(BufReader::new(&input[..])).unwrap();
        let record = &records[0];
        // C `esl_msa_SetAuthor` overwrites: the LAST #=GF AU line wins (verified
        // with `esl-reformat stockholm`).
        assert_eq!(record.msa.author.as_deref(), Some("second author"));
        assert_eq!(record.cutoffs.ga, Some([25.0, 24.5]));
        assert_eq!(record.cutoffs.tc, Some([30.0, 30.0]));
        assert_eq!(record.cutoffs.nc, Some([-1.0, -2.0]));
    }

    #[test]
    fn parses_stockholm_sequence_weights() {
        let input = b"# STOCKHOLM 1.0\n#=GS s1 WT 0.25\n#=GS s2 WT 1.75\ns1 AC\ns2 AC\n//\n";
        let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
        assert_eq!(msas[0].weights.as_deref(), Some(&[0.25, 1.75][..]));

        let input = b"# STOCKHOLM 1.0\n#=GS s1 WT 0.25\ns1 AC\ns2 AC\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("without a weight for s2"));
    }

    #[test]
    fn rejects_stockholm_id_with_extra_fields() {
        let input = b"# STOCKHOLM 1.0\n#=GF ID test with cruft\ns1 AC\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("exactly one name token"));
    }

    #[test]
    fn rejects_duplicate_stockholm_sequence_descriptions() {
        let input = b"# STOCKHOLM 1.0\n#=GS s1 DE first\n#=GS s1 DE second\ns1 AC\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("more than one DE annotation"));
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

    #[test]
    fn rejects_stockholm_duplicate_sequence_row_in_block() {
        let input = b"# STOCKHOLM 1.0\ns1 AC\ns1 GU\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("more than once in the same block"));
    }

    #[test]
    fn rejects_stockholm_multiblock_order_mismatch() {
        let input = b"# STOCKHOLM 1.0\ns1 AC\ns2 AC\n\ns2 GU\ns1 GU\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("out of order"));
    }

    #[test]
    fn rejects_stockholm_multiblock_gc_order_mismatch() {
        let input =
            b"# STOCKHOLM 1.0\ns1 AC\n#=GC RF xx\n#=GC PP_cons 99\n\ns1 GU\n#=GC PP_cons 99\n#=GC RF xx\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("out of order"));
        assert!(err.to_string().contains("#=GC PP_cons"));
    }

    #[test]
    fn rejects_stockholm_multiblock_gr_order_mismatch() {
        let input =
            b"# STOCKHOLM 1.0\ns1 AC\ns2 AC\n#=GR s1 PP 99\n#=GR s2 PP 88\n\ns1 GU\ns2 GU\n#=GR s2 PP 88\n#=GR s1 PP 99\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("out of order"));
        assert!(err.to_string().contains("#=GR s2 PP"));
    }

    #[test]
    fn accepts_stockholm_multiblock_same_order() {
        let input = b"# STOCKHOLM 1.0\ns1 AC\ns2 AC\n\ns1 GU\ns2 GU\n//\n";
        let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
        assert_eq!(msas[0].aseq[0], b"ACGU");
        assert_eq!(msas[0].aseq[1], b"ACGU");
    }

    #[test]
    fn parses_a2m_consensus_and_insert_columns() {
        let input = b">s1 first\nGAATTC\n>s2 second\nGAaATTC\n";
        let mut reader = &input[..];
        let msas = read_a2m_from_reader(&mut reader, "toy".to_string()).unwrap();
        let msa = &msas[0];
        assert_eq!(msa.name, "toy");
        assert_eq!(msa.sqname, ["s1", "s2"]);
        assert_eq!(msa.sqdesc, ["first", "second"]);
        assert_eq!(msa.alen, 7);
        // Insert-column gaps are '.' (C esl-reformat emits "GA.ATTC"), verified
        // against `esl-reformat afa` on the same A2M input.
        assert_eq!(msa.aseq[0], b"GA.ATTC");
        assert_eq!(msa.aseq[1], b"GAaATTC");
        assert_eq!(msa.rf.as_deref(), Some(&b"xx.xxxx"[..]));
    }

    #[test]
    fn rejects_a2m_rows_with_different_consensus_counts() {
        let input = b">s1\nGAATTC\n>s2\nGAaTTC\n";
        let mut reader = &input[..];
        let err = read_a2m_from_reader(&mut reader, "toy".to_string()).unwrap_err();
        assert!(err.to_string().contains("consensus columns"));
    }

    #[test]
    fn full_reader_buckets_tags_and_writer_reflows() {
        // GF specials (ID/AC/DE) + arbitrary XX; GS DE + arbitrary OS; GR SS/PP
        // (input order PP then SS -> writer emits SS then PP); GC SS_cons/RF/MM
        // (writer order SS_cons, ..., RF, MM, then arbitrary). Written in text
        // mode (abc=None) so residues pass through verbatim.
        let input = b"# STOCKHOLM 1.0\n#=GF ID meta\n#=GF AC PF1\n#=GF DE a desc\n#=GF XX extra gf\n#=GS s1 DE seq desc\n#=GS s1 OS extra gs\ns1 ACDEFG\n#=GR s1 PP 999999\n#=GR s1 SS HHHHHH\n#=GC RF xxxxxx\n#=GC SS_cons <<<<<<\n#=GC MM ......\n//\n";
        let msas = read_stockholm_full_from_reader(BufReader::new(&input[..])).unwrap();
        assert_eq!(msas.len(), 1);
        let m = &msas[0];
        assert_eq!(m.name.as_deref(), Some("meta"));
        assert_eq!(m.acc.as_deref(), Some("PF1"));
        assert_eq!(m.desc.as_deref(), Some("a desc"));
        assert_eq!(m.gf, vec![("XX".to_string(), "extra gf".to_string())]);
        assert_eq!(m.sqdesc[0].as_deref(), Some("seq desc"));
        assert_eq!(m.gs[0].0, "OS");
        assert_eq!(m.ss[0].as_deref(), Some(&b"HHHHHH"[..]));
        assert_eq!(m.pp[0].as_deref(), Some(&b"999999"[..]));
        assert_eq!(m.ss_cons.as_deref(), Some(&b"<<<<<<"[..]));
        assert_eq!(m.rf.as_deref(), Some(&b"xxxxxx"[..]));

        let mut out = Vec::new();
        write_stockholm_full(&mut out, m, 200, None).unwrap();
        let s = String::from_utf8(out).unwrap();
        // GF reflow: arbitrary XX after the specials.
        let idx_de = s.find("#=GF DE a desc\n").unwrap();
        let idx_xx = s.find("#=GF XX extra gf\n").unwrap();
        assert!(idx_de < idx_xx);
        // GS: DE group then OS group, each with a blank line after.
        assert!(s.contains("#=GS s1 DE seq desc\n\n#=GS s1 OS extra gs\n\n"));
        // GR reordered: SS before PP.
        let idx_ss = s.find("#=GR s1 SS").unwrap();
        let idx_pp = s.find("#=GR s1 PP").unwrap();
        assert!(idx_ss < idx_pp);
        // GC reordered: SS_cons, then RF, then MM.
        let i_ss = s.find("#=GC SS_cons").unwrap();
        let i_rf = s.find("#=GC RF").unwrap();
        let i_mm = s.find("#=GC MM").unwrap();
        assert!(i_ss < i_rf && i_rf < i_mm);
        assert!(s.ends_with("//\n"));
    }

    #[test]
    fn full_writer_normalizes_gaps_with_alphabet() {
        // Digital-mode write (abc=Some) normalizes '.'/'_' gap chars to '-' in
        // sequences (esl_abc_TextizeN), matching alimask; #=GC RF stays text.
        let input = b"# STOCKHOLM 1.0\ns1 AC.DEF\ns2 AC_DEF\n#=GC RF xx.xxx\n//\n";
        let m = read_stockholm_full_from_reader(BufReader::new(&input[..]))
            .unwrap()
            .remove(0);
        let mut out = Vec::new();
        write_stockholm_full(&mut out, &m, 200, Some(&crate::alphabet::Alphabet::amino())).unwrap();
        let s = String::from_utf8(out).unwrap();
        // Gap chars '.' and '_' normalize to '-'; margin is driven by #=GC RF
        // (maxgc=2 -> margin=8), so names pad to width 7.
        assert!(s.contains("s1      AC-DEF\n"), "{s}");
        assert!(s.contains("s2      AC-DEF\n"), "{s}");
        // RF annotation is not a digitized sequence; its '.' is preserved.
        assert!(s.contains("#=GC RF xx.xxx\n"), "{s}");
    }

    #[test]
    fn full_writer_blockwraps_at_cpl() {
        let input = b"# STOCKHOLM 1.0\ns1 AAAACCCCGG\ns2 AAAACCCCGG\n//\n";
        let m = read_stockholm_full_from_reader(BufReader::new(&input[..]))
            .unwrap()
            .remove(0);
        let mut out = Vec::new();
        write_stockholm_full(&mut out, &m, 5, None).unwrap();
        let s = String::from_utf8(out).unwrap();
        // alen=10, cpl=5 -> two blocks separated by a blank line.
        assert!(
            s.contains("s1 AAAAC\ns2 AAAAC\n\ns1 CCCGG\ns2 CCCGG\n//\n"),
            "{s}"
        );
    }

    #[test]
    fn rejects_stockholm_junk_preamble() {
        let input = b"junk\n# STOCKHOLM 1.0\ns1 AC\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("Expected Stockholm header"));
    }

    #[test]
    fn text_msa_reader_rejects_oversized_input_before_full_allocation() {
        let input = b"# STOCKHOLM 1.0\ns1 AC\n//\n";
        let err = read_text_msa_to_string_with_limit(&mut &input[..], 8).unwrap_err();
        assert!(err.to_string().contains("MSA input exceeds 8 bytes"));
    }
}
