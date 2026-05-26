//! Multiple Sequence Alignment (MSA) I/O.

use crate::alphabet::{Alphabet, Dsq, DSQ_ILLEGAL};
use crate::errors::{HmmerError, HmmerResult};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    let mut text = String::new();
    reader.read_to_string(&mut text).map_err(HmmerError::Io)?;
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
    for idx in 0..=ncons {
        rf.extend(std::iter::repeat(b'.').take(insert_widths[idx]));
        if idx < ncons {
            rf.push(b'x');
        }
    }

    let mut sqname = Vec::with_capacity(records.len());
    let mut sqdesc = Vec::with_capacity(records.len());
    let mut aseq = Vec::with_capacity(records.len());
    for record in records {
        let mut row = Vec::with_capacity(alen);
        for idx in 0..=ncons {
            row.extend_from_slice(&record.inserts[idx]);
            row.extend(
                std::iter::repeat(b'-').take(insert_widths[idx] - record.inserts[idx].len()),
            );
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
    let mut row = 0usize;

    for (line_idx, line) in records {
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
        row += 1;
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

fn parse_phylip_lines(text: &str) -> HmmerResult<(usize, usize, Vec<(usize, &str)>)> {
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

        if trimmed.starts_with("#=GF ID") {
            let fields: Vec<&str> = trimmed[7..].split_whitespace().collect();
            if fields.len() != 1 {
                return Err(HmmerError::Format(format!(
                    "Stockholm #=GF ID annotation must contain exactly one name token, got '{}'",
                    trimmed[7..].trim()
                )));
            }
            name = fields[0].to_string();
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
        } else if trimmed.starts_with("#=GF AU") {
            let line = trimmed[7..].trim();
            if !line.is_empty() {
                author.get_or_insert_with(String::new);
                if let Some(author) = &mut author {
                    if !author.is_empty() {
                        author.push(' ');
                    }
                    author.push_str(line);
                }
            }
        } else if trimmed.starts_with("#=GF GA") {
            cutoffs.ga = Some(parse_stockholm_cutoffs("GA", trimmed[7..].trim())?);
        } else if trimmed.starts_with("#=GF TC") {
            cutoffs.tc = Some(parse_stockholm_cutoffs("TC", trimmed[7..].trim())?);
        } else if trimmed.starts_with("#=GF NC") {
            cutoffs.nc = Some(parse_stockholm_cutoffs("NC", trimmed[7..].trim())?);
        } else if trimmed.starts_with("#=GC RF") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("RF".to_string()),
            )?;
            let rf_str = trimmed[7..].trim();
            match &mut rf {
                Some(existing) => existing.extend_from_slice(rf_str.as_bytes()),
                None => rf = Some(rf_str.as_bytes().to_vec()),
            }
        } else if trimmed.starts_with("#=GC MM") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("MM".to_string()),
            )?;
            let mm_str = trimmed[7..].trim();
            match &mut mm {
                Some(existing) => existing.extend_from_slice(mm_str.as_bytes()),
                None => mm = Some(mm_str.as_bytes().to_vec()),
            }
        } else if trimmed.starts_with("#=GC SS_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("SS_cons".to_string()),
            )?;
            let ss_str = trimmed[12..].trim();
            match &mut ss_cons {
                Some(existing) => existing.extend_from_slice(ss_str.as_bytes()),
                None => ss_cons = Some(ss_str.as_bytes().to_vec()),
            }
        } else if trimmed.starts_with("#=GC SA_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("SA_cons".to_string()),
            )?;
            let sa_str = trimmed[12..].trim();
            match &mut sa_cons {
                Some(existing) => existing.extend_from_slice(sa_str.as_bytes()),
                None => sa_cons = Some(sa_str.as_bytes().to_vec()),
            }
        } else if trimmed.starts_with("#=GC PP_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("PP_cons".to_string()),
            )?;
            let pp_str = trimmed[12..].trim();
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

                if !seq_data.contains_key(&sqname) {
                    seq_order.push(sqname.clone());
                    seq_data.insert(sqname, sqdata.to_vec());
                } else {
                    seq_data.get_mut(&sqname).unwrap().extend_from_slice(sqdata);
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
        assert!(msa.weights.is_none());
        assert_eq!(msa.pp[0].as_deref(), Some(&b"9*"[..]));
        assert_eq!(msa.pp_cons.as_deref(), Some(&b"8*"[..]));
    }

    #[test]
    fn parses_stockholm_author_and_cutoffs() {
        let input = b"# STOCKHOLM 1.0\n#=GF AU first author\n#=GF AU second author\n#=GF GA 25.0 24.5\n#=GF TC 30.0;\n#=GF NC -1.0 -2.0;\ns1 AC\n//\n";
        let records = read_stockholm_preserved_from_reader(BufReader::new(&input[..])).unwrap();
        let record = &records[0];
        assert_eq!(
            record.msa.author.as_deref(),
            Some("first author second author")
        );
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
        assert_eq!(msa.aseq[0], b"GA-ATTC");
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
    fn rejects_stockholm_junk_preamble() {
        let input = b"junk\n# STOCKHOLM 1.0\ns1 AC\n//\n";
        let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
        assert!(err.to_string().contains("Expected Stockholm header"));
    }
}
