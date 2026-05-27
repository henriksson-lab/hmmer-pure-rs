//! HMM file I/O: reading and writing HMMER3 format HMM files.
//! Direct port of p7_hmmfile.c.

use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use crate::alphabet::{Alphabet, AlphabetType};
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;
use crate::output::{fmt_fixed2, fmt_fixed4, fmt_fixed5, fmt_fixed6, fmt_hmm_prob};
use crate::util::cmath::{c_expf_to_f32, c_logf_to_f32};

/// Open an HMM save file and read every HMM contained in it.
///
/// Wrapper over [`read_hmms`] that opens `path` first. Returns all HMMs
/// in the file as a `Vec`; errors propagate I/O and format failures.
pub fn read_hmm_file(path: &Path) -> HmmerResult<Vec<Hmm>> {
    let file = std::fs::File::open(path).map_err(|e| HmmerError::Io(e))?;
    let reader = BufReader::new(file);
    read_hmms(reader)
}

/// Open an HMM save file and read every HMM, auto-dispatching ASCII vs binary
/// from the leading magic bytes instead of relying on the filename.
pub fn read_hmm_file_auto(path: &Path) -> HmmerResult<Vec<Hmm>> {
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        crate::hmmfile_binary::read_binary_hmm_file(path)
    } else {
        read_hmm_file(path)
    }
}

/// Open an HMM save file and read records without enforcing a single ASCII
/// HMMER3 format tag across the whole file. This is only for legacy utility
/// paths that operate record-by-record rather than validating a database.
pub fn read_hmm_file_auto_allow_mixed_formats(path: &Path) -> HmmerResult<Vec<Hmm>> {
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        crate::hmmfile_binary::read_binary_hmm_file(path)
    } else {
        let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
        read_hmms_allow_mixed_formats(BufReader::new(file))
    }
}

/// Open an HMM save file and read only the first HMM record.
pub fn read_first_hmm_file_auto(path: &Path) -> HmmerResult<Hmm> {
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        return crate::hmmfile_binary::read_binary_hmm_file(path)?
            .into_iter()
            .next()
            .ok_or_else(|| HmmerError::Format("No HMM records found".to_string()));
    }
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    read_first_hmm(BufReader::new(file))
}

/// Seek to a record offset from an SSI index and read exactly one HMM record.
pub fn read_hmm_file_record_at(path: &Path, offset: u64) -> HmmerResult<Hmm> {
    let mut file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    file.seek(SeekFrom::Start(offset)).map_err(HmmerError::Io)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(HmmerError::Io)?;
    file.seek(SeekFrom::Start(offset)).map_err(HmmerError::Io)?;
    if crate::hmmfile_binary::is_binary_hmm_magic(u32::from_ne_bytes(magic)) {
        return crate::hmmfile_binary::read_binary_hmm(&mut BufReader::new(file))?
            .ok_or_else(|| HmmerError::Format(format!("No binary HMM record at offset {offset}")));
    }

    let mut reader = BufReader::new(file);
    let mut record = String::new();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(HmmerError::Io)?;
        if n == 0 {
            break;
        }
        record.push_str(&line);
        if line.trim() == "//" {
            break;
        }
    }
    if record.is_empty() {
        return Err(HmmerError::Format(format!(
            "No ASCII HMM record at offset {offset}"
        )));
    }
    if !record.trim_end().ends_with("//") {
        return Err(HmmerError::Format(format!(
            "Unterminated ASCII HMM record at offset {offset}"
        )));
    }
    let mut hmms = read_hmms(BufReader::new(Cursor::new(record.into_bytes())))?;
    if hmms.len() != 1 {
        return Err(HmmerError::Format(format!(
            "Expected one HMM record at offset {offset}, found {}",
            hmms.len()
        )));
    }
    Ok(hmms.remove(0))
}

/// Read all HMMs from an open HMM save file stream (Rust port of `p7_hmmfile_Read`).
///
/// Loops calling `read_one_hmm` until the reader hits EOF, collecting each
/// parsed `Hmm` into a vector. The C entry point reads one HMM at a time via
/// a parser dispatch (`read_asc30hmm` / `read_bin30hmm`); here we expose the
/// "read everything" idiom and a single ASCII parser.
pub fn read_hmms<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Hmm>> {
    let mut hmms = Vec::new();
    let mut lines = reader.lines();
    let mut expected_abc = None;
    let mut expected_format: Option<String> = None;

    loop {
        match read_one_hmm_with_format(&mut lines)? {
            Some((hmm, format_version)) => {
                if let Some(expected) = expected_format.as_deref() {
                    if format_version != expected {
                        return Err(HmmerError::Format(format!(
                            "ASCII HMM file contains mixed HMMER3 format versions: first record is {expected}, record {} is {format_version}",
                            hmms.len() + 1
                        )));
                    }
                } else {
                    expected_format = Some(format_version.clone());
                }
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "ASCII HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }

    Ok(hmms)
}

pub fn read_hmms_allow_mixed_formats<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Hmm>> {
    let mut hmms = Vec::new();
    let mut lines = reader.lines();
    let mut expected_abc = None;

    loop {
        match read_one_hmm_with_format(&mut lines)? {
            Some((hmm, _format_version)) => {
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "ASCII HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }

    Ok(hmms)
}

pub fn read_first_hmm<R: Read>(reader: BufReader<R>) -> HmmerResult<Hmm> {
    let mut lines = reader.lines();
    read_one_hmm_with_format(&mut lines)?
        .map(|(hmm, _format_version)| hmm)
        .ok_or_else(|| HmmerError::Format("No HMM records found".to_string()))
}

/// Read all HMMs from an open stream, auto-dispatching ASCII vs binary from
/// the first four bytes. The prefix is chained back into the selected parser.
pub fn read_hmms_auto<R: Read>(mut reader: BufReader<R>) -> HmmerResult<Vec<Hmm>> {
    let mut prefix = Vec::with_capacity(4);
    while prefix.len() < 4 {
        let mut byte = [0u8; 1];
        match reader.read(&mut byte).map_err(HmmerError::Io)? {
            0 => break,
            1 => prefix.push(byte[0]),
            _ => unreachable!(),
        }
    }

    let is_binary = prefix.len() == 4
        && crate::hmmfile_binary::is_binary_hmm_magic(u32::from_ne_bytes([
            prefix[0], prefix[1], prefix[2], prefix[3],
        ]));
    let chained = Cursor::new(prefix).chain(reader);
    if !is_binary {
        return read_hmms(BufReader::new(chained));
    }

    let mut reader = BufReader::new(chained);
    let mut hmms = Vec::new();
    let mut expected_abc = None;
    loop {
        match crate::hmmfile_binary::read_binary_hmm(&mut reader)? {
            Some(hmm) => {
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "Binary HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }
    Ok(hmms)
}

/// Parse one HMMER3 ASCII HMM record from a line iterator (port of `read_asc30hmm`).
///
/// Handles every header key/value pair (NAME, LENG, ALPH, RF, MM, CONS, CS, MAP,
/// DATE, COM, NSEQ, EFFN, CKSUM, STATS, GA, TC, NC), then reads the alphabet
/// header, an optional COMPO line, node-0 insert/transitions, and the M
/// node blocks of match/insert/transition lines, terminated by `//`.
/// File values are stored as `-ln(p)` and are exponentiated back into
/// probabilities on the fly. Returns `Ok(None)` on clean EOF.
fn read_one_hmm_with_format<B: BufRead>(
    lines: &mut std::io::Lines<B>,
) -> HmmerResult<Option<(Hmm, String)>> {
    // Find the format header line
    let header = loop {
        match lines.next() {
            None => return Ok(None),
            Some(Err(e)) => return Err(HmmerError::Io(e)),
            Some(Ok(line)) => {
                let trimmed = line.trim();
                if trimmed.starts_with("HMMER3/") {
                    break trimmed.to_string();
                }
                if trimmed.starts_with("HMMER2.") || trimmed.starts_with("HMMER2/") {
                    return Err(HmmerError::Format(
                        "HMMER2 ASCII input is intentionally unsupported".to_string(),
                    ));
                }
                // Skip blank lines between HMMs
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    return Err(HmmerError::Format(format!(
                        "Expected HMMER3/ header, got: {}",
                        trimmed
                    )));
                }
            }
        }
    };

    let format_version = parse_hmmer3_ascii_magic(&header)?;
    // Parse header key-value pairs
    let mut name: Option<String> = None;
    let mut acc: Option<String> = None;
    let mut desc: Option<String> = None;
    let mut m: usize = 0;
    let mut max_length: i32 = -1;
    let mut abc_type = AlphabetType::Unknown;
    let mut rf_flag = false;
    let mut mm_flag = false;
    let mut cons_flag = false;
    let mut cs_flag = false;
    let mut map_flag = false;
    let mut nseq: i32 = -1;
    let mut eff_nseq: f32 = -1.0;
    let mut ctime: Option<String> = None;
    let mut comlog: Option<String> = None;
    let mut checksum: u32 = 0;
    // checksum presence is tracked via P7H_CHKSUM flag
    let mut evparam = [EVPARAM_UNSET; NEVPARAM];
    let mut cutoff = [CUTOFF_UNSET; NCUTOFFS];
    let mut flags: u32 = 0;
    let mut stat_msv = false;
    let mut stat_viterbi = false;
    let mut stat_forward = false;

    // Read header lines until "HMM" line
    loop {
        let line = lines
            .next()
            .ok_or_else(|| HmmerError::Format("Unexpected EOF in HMM header".to_string()))?
            .map_err(HmmerError::Io)?;
        let trimmed = line.trim();

        if trimmed.starts_with("HMM ") || trimmed == "HMM" {
            validate_alphabet_header(trimmed, abc_type)?;
            break;
        }

        let (key, value) = match trimmed.split_once(char::is_whitespace) {
            Some((k, v)) => (k, v.trim()),
            None => continue,
        };

        match key {
            "NAME" => name = Some(value.to_string()),
            "ACC" => {
                acc = Some(value.to_string());
                flags |= P7H_ACC;
            }
            "DESC" => {
                desc = Some(value.to_string());
                flags |= P7H_DESC;
            }
            "LENG" => {
                m = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad LENG".to_string()))?;
                if m == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid model length {value} on LENG line"
                    )));
                }
            }
            "MAXL" => {
                max_length = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad MAXL".to_string()))?;
                if max_length == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid max length {value} on MAXL line"
                    )));
                }
            }
            "ALPH" => {
                abc_type = if value.eq_ignore_ascii_case("amino") {
                    AlphabetType::Amino
                } else if value.eq_ignore_ascii_case("DNA") {
                    AlphabetType::Dna
                } else if value.eq_ignore_ascii_case("RNA") {
                    AlphabetType::Rna
                } else {
                    return Err(HmmerError::Format(format!("Unknown alphabet: {}", value)));
                };
            }
            "RF" => rf_flag = parse_hmm_yes_no("RF", value)?,
            "MM" => mm_flag = parse_hmm_yes_no("MM", value)? && format_version == "3f",
            "CONS" => cons_flag = parse_hmm_yes_no("CONS", value)?,
            "CS" => cs_flag = parse_hmm_yes_no("CS", value)?,
            "MAP" => map_flag = parse_hmm_yes_no("MAP", value)?,
            "DATE" => ctime = Some(value.to_string()),
            "COM" => {
                // COM lines may be numbered: "COM   [1] hmmbuild ..."
                let cmd = if value.starts_with('[') {
                    value
                        .split_once(']')
                        .map(|(_, r)| r.trim())
                        .unwrap_or(value)
                } else {
                    value
                };
                match &mut comlog {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(cmd);
                    }
                    None => comlog = Some(cmd.to_string()),
                }
            }
            "NSEQ" => {
                nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad NSEQ".to_string()))?;
                if nseq == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid nseq on NSEQ line: should be integer, not {value}"
                    )));
                }
            }
            "EFFN" => {
                eff_nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad EFFN".to_string()))?;
                if eff_nseq <= 0.0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid eff_nseq on EFFN line: should be a real number, not {value}"
                    )));
                }
            }
            "CKSUM" => {
                checksum = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad CKSUM".to_string()))?;
                flags |= P7H_CHKSUM;
            }
            "STATS" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if format_version == "3a" {
                    // HMMER3/a (reverse compatibility): 3-token form
                    //   "LOCAL VLAMBDA <v>", "LOCAL VMU <v>", "LOCAL FTAU <v>"
                    if parts.len() < 3 {
                        return Err(HmmerError::Format(
                            "Too few fields on STATS line".to_string(),
                        ));
                    }
                    if !parts[0].eq_ignore_ascii_case("LOCAL") {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 2",
                            parts[0]
                        )));
                    }
                    let v: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    if parts[1].eq_ignore_ascii_case("VLAMBDA") {
                        evparam[P7_MLAMBDA] = v;
                        evparam[P7_VLAMBDA] = v;
                        evparam[P7_FLAMBDA] = v;
                        stat_msv = true;
                    } else if parts[1].eq_ignore_ascii_case("VMU") {
                        evparam[P7_MMU] = v;
                        evparam[P7_VMU] = v;
                        stat_viterbi = true;
                    } else if parts[1].eq_ignore_ascii_case("FTAU") {
                        evparam[P7_FTAU] = v;
                        stat_forward = true;
                    } else {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 3",
                            parts[1]
                        )));
                    }
                } else {
                    // HMMER3/b+ : 4-token form "LOCAL MSV -6.4582 0.72049"
                    if parts.len() < 4 {
                        return Err(HmmerError::Format(
                            "Too few fields on STATS line".to_string(),
                        ));
                    }
                    if !parts[0].eq_ignore_ascii_case("LOCAL") {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 2",
                            parts[0]
                        )));
                    }
                    let v1: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    let v2: f32 = parts[3]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    if parts[1].eq_ignore_ascii_case("MSV") {
                        evparam[P7_MMU] = v1;
                        evparam[P7_MLAMBDA] = v2;
                        stat_msv = true;
                    } else if parts[1].eq_ignore_ascii_case("VITERBI") {
                        evparam[P7_VMU] = v1;
                        evparam[P7_VLAMBDA] = v2;
                        stat_viterbi = true;
                    } else if parts[1].eq_ignore_ascii_case("FORWARD") {
                        evparam[P7_FTAU] = v1;
                        evparam[P7_FLAMBDA] = v2;
                        stat_forward = true;
                    } else {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 3",
                            parts[1]
                        )));
                    }
                }
            }
            "GA" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_GA1] = parse_cutoff_value("GA", v)?;
                } else {
                    return Err(HmmerError::Format("Missing GA cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_GA2] = parse_cutoff_value("GA", v)?;
                } else {
                    cutoff[P7_GA2] = cutoff[P7_GA1];
                }
                flags |= P7H_GA;
            }
            "TC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_TC1] = parse_cutoff_value("TC", v)?;
                } else {
                    return Err(HmmerError::Format("Missing TC cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_TC2] = parse_cutoff_value("TC", v)?;
                } else {
                    cutoff[P7_TC2] = cutoff[P7_TC1];
                }
                flags |= P7H_TC;
            }
            "NC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_NC1] = parse_cutoff_value("NC", v)?;
                } else {
                    return Err(HmmerError::Format("Missing NC cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_NC2] = parse_cutoff_value("NC", v)?;
                } else {
                    cutoff[P7_NC2] = cutoff[P7_NC1];
                }
                flags |= P7H_NC;
            }
            _ => {} // Ignore unknown header fields
        }
    }

    let name = name.ok_or_else(|| HmmerError::Format("Missing NAME in HMM".to_string()))?;
    if m == 0 {
        return Err(HmmerError::Format("Missing or zero LENG".to_string()));
    }
    if abc_type == AlphabetType::Unknown {
        return Err(HmmerError::Format("Missing ALPH in HMM".to_string()));
    }
    if stat_msv || stat_viterbi || stat_forward {
        if !(stat_msv && stat_viterbi && stat_forward) {
            return Err(HmmerError::Format(
                "Incomplete STATS block in HMM header".to_string(),
            ));
        }
        flags |= P7H_STATS;
    }

    let abc = crate::alphabet::Alphabet::new(abc_type);
    let k = abc.k;

    // Skip the transition label line ("m->m m->i ...")
    // Skip the transition label line ("m->m m->i ...")
    lines
        .next()
        .ok_or_else(|| HmmerError::Format("Missing transition label line".to_string()))?
        .map_err(HmmerError::Io)?;

    // Create the HMM
    let mut hmm = Hmm::new(m, abc_type, k);
    hmm.name = name;
    hmm.acc = acc;
    hmm.desc = desc;
    hmm.nseq = nseq;
    hmm.eff_nseq = eff_nseq;
    hmm.max_length = max_length;
    hmm.ctime = ctime;
    hmm.comlog = comlog;
    hmm.checksum = checksum;
    hmm.evparam = evparam;
    hmm.cutoff = cutoff;
    hmm.flags = flags;

    // Initialize optional annotation arrays
    if rf_flag {
        let mut rf = vec![b' '; m + 2];
        rf[0] = b' ';
        hmm.rf = Some(rf);
        hmm.flags |= P7H_RF;
    }
    if mm_flag {
        let mut mm = vec![b' '; m + 2];
        mm[0] = b' ';
        hmm.mm = Some(mm);
        hmm.flags |= P7H_MMASK;
    }
    if cons_flag {
        let mut cons = vec![b' '; m + 2];
        cons[0] = b' ';
        hmm.consensus = Some(cons);
        hmm.flags |= P7H_CONS;
    }
    if cs_flag {
        let mut cs = vec![b' '; m + 2];
        cs[0] = b' ';
        hmm.cs = Some(cs);
        hmm.flags |= P7H_CS;
    }
    if map_flag {
        hmm.map = Some(vec![0i32; m + 1]);
        hmm.flags |= P7H_MAP;
    }

    // Read COMPO line (optional, node 0 match emissions)
    let compo_line = lines
        .next()
        .ok_or_else(|| HmmerError::Format("Missing COMPO/insert line".to_string()))?
        .map_err(HmmerError::Io)?;

    let compo_trimmed = compo_line.trim();
    if compo_trimmed.starts_with("COMPO") {
        // Parse composition values
        let parts: Vec<&str> = compo_trimmed.split_whitespace().collect();
        if parts.len() < k + 1 {
            return Err(HmmerError::Format(format!(
                "COMPO line has too few fields: expected {k}, got {}",
                parts.len().saturating_sub(1)
            )));
        }
        for i in 0..k.min(MAXABET) {
            hmm.compo[i] = parse_hmm_value(parts[i + 1])?;
        }
        hmm.flags |= P7H_COMPO;

        // Read node 0 insert emissions
        let ins_line = lines
            .next()
            .ok_or_else(|| HmmerError::Format("Missing node 0 insert line".to_string()))?
            .map_err(HmmerError::Io)?;
        parse_emission_line(&ins_line, k, &mut hmm.ins[0])?;
    } else {
        // No COMPO line — this line IS the node 0 insert emissions
        parse_emission_line(compo_trimmed, k, &mut hmm.ins[0])?;
    }

    // Read node 0 transitions
    let trans_line = lines
        .next()
        .ok_or_else(|| HmmerError::Format("Missing node 0 transition line".to_string()))?
        .map_err(HmmerError::Io)?;
    parse_transition_line(&trans_line, &mut hmm.t[0])?;

    // Read nodes 1..M
    for node in 1..=m {
        // Match emission line: "  k  <K values> <map> <cons> <rf> <mm/cs>"
        let match_line = lines
            .next()
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} match line", node)))?
            .map_err(HmmerError::Io)?;
        let parts: Vec<&str> = match_line.split_whitespace().collect();

        // Parse node number (first field)
        let node_num: usize = parts
            .first()
            .ok_or_else(|| HmmerError::Format("Empty match line".to_string()))?
            .parse()
            .map_err(|_| HmmerError::Format("Bad node number".to_string()))?;
        if node_num != node {
            return Err(HmmerError::Format(format!(
                "Expected node {}, got {}",
                node, node_num
            )));
        }

        // Parse K emission values
        if parts.len() < k + 1 {
            return Err(HmmerError::Format(format!(
                "Node {} match emission line has too few fields",
                node
            )));
        }
        for i in 0..k {
            hmm.mat[node][i] = parse_hmm_value(parts[i + 1])?;
        }

        let mut annot_idx = k + 1;
        let map_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing MAP column"))
        })?;
        annot_idx += 1;
        if map_flag {
            if let Some(map) = &mut hmm.map {
                map[node] = map_val
                    .parse()
                    .map_err(|_| HmmerError::Format(format!("Bad MAP value: {map_val}")))?;
            }
        }

        let has_cons_column = matches!(format_version.as_str(), "3e" | "3f");
        if has_cons_column {
            let cons_val = parts.get(annot_idx).ok_or_else(|| {
                HmmerError::Format(format!("Node {node} match line missing CONS column"))
            })?;
            annot_idx += 1;
            if cons_flag {
                if let Some(cons) = &mut hmm.consensus {
                    cons[node] = annotation_byte("CONS", cons_val)?;
                }
            }
        }

        let rf_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing RF column"))
        })?;
        annot_idx += 1;
        if rf_flag {
            if let Some(rf) = &mut hmm.rf {
                rf[node] = annotation_byte("RF", rf_val)?;
            }
        }

        if format_version == "3f" {
            let mm_val = parts.get(annot_idx).ok_or_else(|| {
                HmmerError::Format(format!("Node {node} match line missing MM column"))
            })?;
            annot_idx += 1;
            if mm_flag {
                if let Some(mm) = &mut hmm.mm {
                    mm[node] = annotation_byte("MM", mm_val)?;
                }
            }
        }

        let cs_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing CS column"))
        })?;
        if cs_flag {
            if let Some(cs) = &mut hmm.cs {
                cs[node] = annotation_byte("CS", cs_val)?;
            }
        }

        // Insert emission line
        let ins_line = lines
            .next()
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} insert line", node)))?
            .map_err(HmmerError::Io)?;
        parse_emission_line(&ins_line, k, &mut hmm.ins[node])?;

        // Transition line
        let trans_line = lines
            .next()
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} transition line", node)))?
            .map_err(HmmerError::Io)?;
        parse_transition_line(&trans_line, &mut hmm.t[node])?;
    }

    // Read end-of-record marker "//"
    let mut saw_terminator = false;
    loop {
        match lines.next() {
            None => break,
            Some(Err(e)) => return Err(HmmerError::Io(e)),
            Some(Ok(line)) => {
                let trimmed = line.trim();
                if trimmed == "//" {
                    saw_terminator = true;
                    break;
                } else if !trimmed.is_empty() {
                    return Err(HmmerError::Format(format!(
                        "Expected end-of-record marker //, got: {trimmed}"
                    )));
                }
            }
        }
    }
    if !saw_terminator {
        return Err(HmmerError::Format(
            "Unexpected EOF before HMM end-of-record marker //".to_string(),
        ));
    }

    // Convert from -ln(prob) to probability
    // In the file, values are stored as -ln(p). Convert to p.
    for node in 0..=m {
        for i in 0..k {
            hmm.mat[node][i] = c_expf_to_f32(-hmm.mat[node][i]);
            hmm.ins[node][i] = c_expf_to_f32(-hmm.ins[node][i]);
        }
        for i in 0..NTRANSITIONS {
            if hmm.t[node][i] != f32::INFINITY {
                hmm.t[node][i] = c_expf_to_f32(-hmm.t[node][i]);
            } else {
                hmm.t[node][i] = 0.0;
            }
        }
    }

    // Convert compo from -ln(prob) to prob
    for i in 0..k.min(MAXABET) {
        if hmm.compo[i] != COMPO_UNSET {
            hmm.compo[i] = c_expf_to_f32(-hmm.compo[i]);
        }
    }

    Ok(Some((hmm, format_version)))
}

fn parse_hmmer3_ascii_magic(header: &str) -> HmmerResult<String> {
    let tag = header.split_whitespace().next().unwrap_or("");
    let version = match tag {
        "HMMER3/a" => "3a",
        "HMMER3/b" => "3b",
        "HMMER3/c" => "3c",
        "HMMER3/d" => "3d",
        "HMMER3/e" => "3e",
        "HMMER3/f" => "3f",
        _ => {
            return Err(HmmerError::Format(format!(
                "Unsupported HMMER3 ASCII magic tag: {tag}"
            )))
        }
    };
    Ok(version.to_string())
}

fn parse_hmm_yes_no(key: &str, value: &str) -> HmmerResult<bool> {
    if value.eq_ignore_ascii_case("yes") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("no") {
        Ok(false)
    } else {
        Err(HmmerError::Format(format!(
            "Bad {key} value in HMM header: expected yes or no, got {value}"
        )))
    }
}

/// Parse one HMM value: a float, or `"*"` meaning zero probability (stored as `+inf` in -ln space).
fn parse_hmm_value(s: &str) -> HmmerResult<f32> {
    if s == "*" {
        Ok(f32::INFINITY)
    } else {
        s.parse()
            .map_err(|_| HmmerError::Format(format!("Bad HMM probability value: {s}")))
    }
}

fn parse_cutoff_value(key: &str, s: &str) -> HmmerResult<f32> {
    s.trim_end_matches(';')
        .parse()
        .map_err(|_| HmmerError::Format(format!("Bad {key} cutoff value: {s}")))
}

fn validate_alphabet_header(line: &str, abc_type: AlphabetType) -> HmmerResult<()> {
    if abc_type == AlphabetType::Unknown {
        return Err(HmmerError::Format(
            "HMM alphabet header appeared before ALPH".to_string(),
        ));
    }
    let abc = Alphabet::new(abc_type);
    let fields: Vec<&str> = line.split_whitespace().skip(1).collect();
    if fields.len() != abc.k {
        return Err(HmmerError::Format(format!(
            "HMM alphabet header has {} symbols, expected {}",
            fields.len(),
            abc.k
        )));
    }
    for (idx, (got, expected)) in fields.iter().zip(abc.sym.iter()).enumerate() {
        if got.len() != 1 || got.as_bytes()[0] != *expected {
            return Err(HmmerError::Format(format!(
                "HMM alphabet header symbol {} is {}, expected {}",
                idx + 1,
                got,
                *expected as char
            )));
        }
    }
    Ok(())
}

fn annotation_byte(label: &str, s: &str) -> HmmerResult<u8> {
    s.as_bytes()
        .first()
        .copied()
        .ok_or_else(|| HmmerError::Format(format!("Empty {label} annotation")))
}

/// Parse an emission row of `K` whitespace-separated `-ln(p)` values into `values`.
fn parse_emission_line(line: &str, k: usize, values: &mut [f32]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < k {
        return Err(HmmerError::Format(format!(
            "Emission line has too few fields: expected {k}, got {}",
            parts.len()
        )));
    }
    for i in 0..k {
        values[i] = parse_hmm_value(parts[i])?;
    }
    Ok(())
}

/// Parse a transition row of `NTRANSITIONS` whitespace-separated `-ln(p)` values.
fn parse_transition_line(line: &str, values: &mut [f32; NTRANSITIONS]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < NTRANSITIONS {
        return Err(HmmerError::Format(format!(
            "Transition line has too few fields: expected {NTRANSITIONS}, got {}",
            parts.len()
        )));
    }
    for i in 0..NTRANSITIONS {
        values[i] = parse_hmm_value(parts[i])?;
    }
    Ok(())
}

/// Write a profile HMM as a HMMER3 ASCII save file (port of `p7_hmmfile_WriteASCII`).
///
/// Emits the HMMER3 header block (NAME/ACC/DESC/LENG/MAXL/ALPH, annotation
/// flag lines, DATE/NSEQ/EFFN/CKSUM, STATS LOCAL MSV/VITERBI/FORWARD), then
/// the alphabet line, transition label legend, optional COMPO line, node-0
/// insert/transition rows, and one match/insert/transition triplet per node
/// 1..M, terminated by `//`. Probabilities are encoded as `-ln(p)` with `*`
/// for zero, matching the C writer's `logf`-based output exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HmmAsciiFormat {
    Hmmer3a,
    Hmmer3b,
    Hmmer3c,
    Hmmer3d,
    Hmmer3e,
    Hmmer3f,
}

impl HmmAsciiFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "3/a" => Some(Self::Hmmer3a),
            "3/b" => Some(Self::Hmmer3b),
            "3/c" => Some(Self::Hmmer3c),
            "3/d" => Some(Self::Hmmer3d),
            "3/e" => Some(Self::Hmmer3e),
            "3/f" => Some(Self::Hmmer3f),
            _ => None,
        }
    }

    fn code(self) -> char {
        match self {
            Self::Hmmer3a => 'a',
            Self::Hmmer3b => 'b',
            Self::Hmmer3c => 'c',
            Self::Hmmer3d => 'd',
            Self::Hmmer3e => 'e',
            Self::Hmmer3f => 'f',
        }
    }

    fn is_at_least(self, other: Self) -> bool {
        self >= other
    }
}

pub fn write_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    write_hmm_with_format(w, hmm, HmmAsciiFormat::Hmmer3f)
}

pub fn write_hmm_with_format<W: std::io::Write>(
    w: &mut W,
    hmm: &Hmm,
    format: HmmAsciiFormat,
) -> HmmerResult<()> {
    let k = hmm.abc_k;

    if format == HmmAsciiFormat::Hmmer3f {
        writeln!(w, "HMMER3/f [3.4 | Aug 2023]").map_err(HmmerError::Io)?;
    } else {
        writeln!(
            w,
            "HMMER3/{} [3.4 | Aug 2023; reverse compatibility mode]",
            format.code()
        )
        .map_err(HmmerError::Io)?;
    }
    writeln!(w, "NAME  {}", hmm.name).map_err(HmmerError::Io)?;
    if let Some(ref acc) = hmm.acc {
        writeln!(w, "ACC   {}", acc).map_err(HmmerError::Io)?;
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(w, "DESC  {}", desc).map_err(HmmerError::Io)?;
    }
    writeln!(w, "LENG  {}", hmm.m).map_err(HmmerError::Io)?;
    if format.is_at_least(HmmAsciiFormat::Hmmer3c) && hmm.max_length > 0 {
        writeln!(w, "MAXL  {}", hmm.max_length).map_err(HmmerError::Io)?;
    }
    let alph = match hmm.abc_type {
        AlphabetType::Amino => "amino",
        AlphabetType::Dna => "DNA",
        AlphabetType::Rna => "RNA",
        AlphabetType::Unknown => "unknown",
    };
    writeln!(w, "ALPH  {}", alph).map_err(HmmerError::Io)?;
    writeln!(
        w,
        "RF    {}",
        if hmm.flags & P7H_RF != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    if format.is_at_least(HmmAsciiFormat::Hmmer3f) {
        writeln!(
            w,
            "MM    {}",
            if hmm.flags & P7H_MMASK != 0 {
                "yes"
            } else {
                "no"
            }
        )
        .map_err(HmmerError::Io)?;
    }
    if format.is_at_least(HmmAsciiFormat::Hmmer3e) {
        writeln!(
            w,
            "CONS  {}",
            if hmm.flags & P7H_CONS != 0 {
                "yes"
            } else {
                "no"
            }
        )
        .map_err(HmmerError::Io)?;
    }
    writeln!(
        w,
        "CS    {}",
        if hmm.flags & P7H_CS != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    writeln!(
        w,
        "MAP   {}",
        if hmm.flags & P7H_MAP != 0 {
            "yes"
        } else {
            "no"
        }
    )
    .map_err(HmmerError::Io)?;
    if let Some(ref ctime) = hmm.ctime {
        writeln!(w, "DATE  {}", ctime).map_err(HmmerError::Io)?;
    }
    if let Some(ref comlog) = hmm.comlog {
        for (idx, command) in comlog.lines().enumerate() {
            writeln!(w, "COM   [{}] {}", idx + 1, command).map_err(HmmerError::Io)?;
        }
    }
    if hmm.nseq >= 0 {
        writeln!(w, "NSEQ  {}", hmm.nseq).map_err(HmmerError::Io)?;
    }
    if hmm.eff_nseq >= 0.0 {
        writeln!(w, "EFFN  {}", fmt_fixed6(hmm.eff_nseq as f64)).map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_CHKSUM != 0 {
        writeln!(w, "CKSUM {}", hmm.checksum).map_err(HmmerError::Io)?;
    }
    // C (p7_hmmfile.c:546-554): nucleic-acid models emit a single cutoff value;
    // only amino emits both the per-sequence and per-domain cutoffs.
    let nucleic = matches!(hmm.abc_type, AlphabetType::Dna | AlphabetType::Rna);
    if hmm.flags & P7H_GA != 0 {
        if nucleic {
            writeln!(w, "GA    {}", fmt_fixed2(hmm.cutoff[P7_GA1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "GA    {} {}",
                fmt_fixed2(hmm.cutoff[P7_GA1] as f64),
                fmt_fixed2(hmm.cutoff[P7_GA2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_TC != 0 {
        if nucleic {
            writeln!(w, "TC    {}", fmt_fixed2(hmm.cutoff[P7_TC1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "TC    {} {}",
                fmt_fixed2(hmm.cutoff[P7_TC1] as f64),
                fmt_fixed2(hmm.cutoff[P7_TC2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_NC != 0 {
        if nucleic {
            writeln!(w, "NC    {}", fmt_fixed2(hmm.cutoff[P7_NC1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "NC    {} {}",
                fmt_fixed2(hmm.cutoff[P7_NC1] as f64),
                fmt_fixed2(hmm.cutoff[P7_NC2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_STATS != 0 {
        if format == HmmAsciiFormat::Hmmer3a {
            writeln!(
                w,
                "STATS LOCAL     VLAMBDA {}",
                fmt_fixed6(hmm.evparam[P7_MLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL         VMU {}",
                fmt_fixed6(hmm.evparam[P7_MMU] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL        FTAU {}",
                fmt_fixed6(hmm.evparam[P7_FTAU] as f64)
            )
            .map_err(HmmerError::Io)?;
        } else {
            // C: fprintf(fp, "STATS LOCAL MSV      %8.4f %8.5f\n", ...)
            // (label + spaces, then a width-8 mu, one space, width-8 lambda).
            // `fmt_fixed4`/`fmt_fixed5` reproduce C's `%.4f`/`%.5f`; right-
            // justifying in width 8 reproduces the `%8.4f`/`%8.5f` field width.
            writeln!(
                w,
                "STATS LOCAL MSV      {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_MMU] as f64),
                fmt_fixed5(hmm.evparam[P7_MLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL VITERBI  {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_VMU] as f64),
                fmt_fixed5(hmm.evparam[P7_VLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL FORWARD  {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_FTAU] as f64),
                fmt_fixed5(hmm.evparam[P7_FLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }

    // Alphabet header
    let abc = crate::alphabet::Alphabet::new(hmm.abc_type);
    write!(w, "HMM     ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, "     {}   ", abc.sym[i] as char).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Transition label line
    writeln!(
        w,
        "            m->m     m->i     m->d     i->m     i->i     d->m     d->d"
    )
    .map_err(HmmerError::Io)?;

    // COMPO line
    if hmm.flags & P7H_COMPO != 0 {
        write!(w, "  COMPO ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.compo[i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    // Node 0 insert emissions
    write!(w, "        ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, " {}", fmt_prob(hmm.ins[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Node 0 transitions
    write!(w, "        ").map_err(HmmerError::Io)?;
    for i in 0..NTRANSITIONS {
        write!(w, " {}", fmt_prob(hmm.t[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Nodes 1..M
    for node in 1..=hmm.m {
        // Match emission line: C writes " %6d " (leading + trailing space).
        write!(w, " {:>6} ", node).map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.mat[node][i])).map_err(HmmerError::Io)?;
        }
        if let Some(ref map) = hmm.map {
            write!(w, " {:>6}", map[node]).map_err(HmmerError::Io)?;
        } else {
            write!(w, " {:>6}", "-").map_err(HmmerError::Io)?;
        }
        let cons_ch = hmm
            .consensus
            .as_ref()
            .map(|cons| cons[node] as char)
            .unwrap_or('-');
        let rf_ch = hmm.rf.as_ref().map(|rf| rf[node] as char).unwrap_or('-');
        let mm_ch = hmm.mm.as_ref().map(|mm| mm[node] as char).unwrap_or('-');
        let cs_ch = hmm.cs.as_ref().map(|cs| cs[node] as char).unwrap_or('-');
        if format.is_at_least(HmmAsciiFormat::Hmmer3e) {
            write!(w, " {}", cons_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", rf_ch).map_err(HmmerError::Io)?;
        if format.is_at_least(HmmAsciiFormat::Hmmer3f) {
            write!(w, " {}", mm_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", cs_ch).map_err(HmmerError::Io)?;
        writeln!(w).map_err(HmmerError::Io)?;

        // Insert emission line
        write!(w, "        ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.ins[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Transition line
        write!(w, "        ").map_err(HmmerError::Io)?;
        for i in 0..NTRANSITIONS {
            write!(w, " {}", fmt_prob(hmm.t[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    writeln!(w, "//").map_err(HmmerError::Io)?;
    Ok(())
}

/// Format a probability `p` as `-ln(p)` (or `*` if zero) using single-precision
/// `logf`, matching the C HMMER ASCII writer's field width and digits.
///
/// Mirrors `printprob(fp, 8, p)` from `hmmer/src/p7_hmmfile.c:2091`:
///   - `p == 0.0` → `" %*s"` with `fieldwidth=8` and `"*"` → `"       *"` (7 spaces + `*`)
///   - `p == 1.0` → `" %*.5f"` with `fieldwidth=8` and `0.0` → `" 0.00000"` (8 chars)
///   - else       → `" %*.5f"` with `fieldwidth=8` and `-logf(p)` → 8-char float string
///
/// Returns an 8-wide string; the caller prepends one space via `" {}"` to
/// reproduce C's `fprintf(fp, " %*.5f", 8, ...)` = 9 bytes per probability value.
fn fmt_prob(p: f32) -> String {
    // C's `printprob` uses `%8s` / `%8.5f` (8-wide field).
    if p <= 0.0 {
        // C: fprintf(fp, " %*s", 8, "*") -> "       *" (7 spaces + '*', 8 chars)
        "       *".to_string()
    } else if p == 1.0 {
        // C: fprintf(fp, " %*.5f", 8, 0.0) -> " 0.00000" (8 chars)
        fmt_hmm_prob(0.0)
    } else {
        // HMMER's C writer uses logf(), not double-precision log().
        // fmt_hmm_prob is now %8.5f, directly matching C's fieldwidth=8.
        fmt_hmm_prob(-c_logf_to_f32(p) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};
    use std::path::Path;

    struct OneByteReader {
        data: Cursor<Vec<u8>>,
    }

    impl std::io::Read for OneByteReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            let mut byte = [0u8; 1];
            let n = self.data.read(&mut byte)?;
            if n == 1 {
                buf[0] = byte[0];
            }
            Ok(n)
        }
    }

    fn read_hmms_from_str(s: &str) -> HmmerResult<Vec<Hmm>> {
        read_hmms(BufReader::new(Cursor::new(s.as_bytes())))
    }

    fn minimal_hmm(name: &str, abc_type: AlphabetType) -> Hmm {
        let abc = Alphabet::new(abc_type);
        let mut hmm = Hmm::new(1, abc_type, abc.k);
        hmm.name = name.to_string();
        hmm
    }

    #[test]
    fn test_read_20aa_hmm() {
        let hmms = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap();
        assert_eq!(hmms.len(), 1);
        let hmm = &hmms[0];
        assert_eq!(hmm.name, "test");
        assert_eq!(hmm.m, 20);
        assert_eq!(hmm.abc_type, AlphabetType::Amino);
        assert_eq!(hmm.abc_k, 20);
        assert_eq!(hmm.nseq, 10);
        assert!((hmm.eff_nseq - 1.958008).abs() < 1e-5);
        assert!(hmm.flags & P7H_STATS != 0);
        assert!(hmm.flags & P7H_RF != 0);
        assert!(hmm.flags & P7H_CONS != 0);
        assert!(hmm.flags & P7H_MAP != 0);

        // Check that match emissions were converted from -ln(p) to probability
        // Node 1 first value (A emission) was 0.33153 in file -> exp(-0.33153) ≈ 0.7182
        assert!(hmm.mat[1][0] > 0.0 && hmm.mat[1][0] < 1.0);

        // Check E-value params
        assert!((hmm.evparam[P7_MMU] - (-6.4582)).abs() < 1e-3);
        assert!((hmm.evparam[P7_MLAMBDA] - 0.72049).abs() < 1e-4);
    }

    #[test]
    fn test_read_fn3_hmm() {
        let hmms = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        )))
        .unwrap();
        assert_eq!(hmms.len(), 1);
        let hmm = &hmms[0];
        assert_eq!(hmm.name, "fn3");
        assert!(hmm.m > 50); // fn3 is a medium-sized domain
    }

    #[test]
    fn auto_reader_detects_binary_hmm_from_short_stream_reads() {
        let hmm = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        )))
        .unwrap()
        .remove(0);
        let mut bytes = Vec::new();
        crate::hmmfile_binary::write_binary_hmm(&mut bytes, &hmm).unwrap();

        let reader = OneByteReader {
            data: Cursor::new(bytes),
        };
        let hmms = read_hmms_auto(BufReader::new(reader)).unwrap();
        assert_eq!(hmms.len(), 1);
        assert_eq!(hmms[0].name, "fn3");
        assert_eq!(hmms[0].m, 86);
    }

    #[test]
    fn binary_annotation_arrays_use_space_sentinel_at_index_0_like_c() {
        // C HMMER's binary writer (`p7_hmmfile_WriteBinary`, p7_hmmfile.c:1037-
        // 1041) emits the raw `M+2`-byte annotation arrays verbatim. Its ASCII
        // reader (`read_asc30hmm`) sets index 0 of rf/mm/consensus/cs to ' '
        // (0x20). Verified empirically against the C `hmmconvert -b` output for
        // fn3: the CONS and CS arrays are byte-identical to Rust's, with a 0x20
        // sentinel at index 0 (NOT a NUL). The audit's F4 claim that C writes a
        // NUL here does not reproduce against the HMMER 3.4 build; C writes a
        // space. This test locks the confirmed C-parity sentinel in.
        let hmm = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        )))
        .unwrap()
        .remove(0);

        // ASCII reader matches C: index 0 == ' ' (0x20).
        let cons = hmm.consensus.as_ref().expect("fn3 has CONS yes");
        assert_eq!(cons[0], b' ', "CONS index 0 must be space (matches C .h3m)");
        let cs = hmm.cs.as_ref().expect("fn3 has CS yes");
        assert_eq!(cs[0], b' ', "CS index 0 must be space (matches C .h3m)");

        // The binary writer must emit those sentinels verbatim. Round-trip
        // through the binary reader (which preserves the bytes verbatim).
        let mut bytes = Vec::new();
        crate::hmmfile_binary::write_binary_hmm(&mut bytes, &hmm).unwrap();
        let rt = crate::hmmfile_binary::read_binary_hmm(&mut Cursor::new(bytes))
            .unwrap()
            .unwrap();
        assert_eq!(rt.consensus.as_ref().unwrap()[0], b' ');
        assert_eq!(rt.cs.as_ref().unwrap()[0], b' ');
    }

    #[test]
    fn rejects_hmm_alphabet_header_that_disagrees_with_alph() {
        let mut text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        ))
        .unwrap();
        text = text.replacen("HMM          A        C", "HMM          C        A", 1);

        let err = read_hmms_from_str(&text).unwrap_err();
        assert!(err.to_string().contains("HMM alphabet header symbol 1"));
    }

    #[test]
    fn rejects_non_exact_hmmer3_magic_tags() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        ))
        .unwrap()
        .replacen("HMMER3/e", "HMMER3/foo", 1);

        let err = read_hmms_from_str(&text).unwrap_err();
        assert!(
            matches!(err, HmmerError::Format(msg) if msg.contains("Unsupported HMMER3 ASCII magic tag"))
        );
    }

    #[test]
    fn parses_alphabet_and_stats_tokens_case_insensitively() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        ))
        .unwrap()
        .replacen("ALPH  amino", "ALPH  AMINO", 1)
        .replacen("STATS LOCAL MSV", "STATS local mSv", 1)
        .replacen("STATS LOCAL VITERBI", "STATS LoCaL vItErBi", 1)
        .replacen("STATS LOCAL FORWARD", "STATS LOCAL forward", 1);

        let hmm = read_hmms_from_str(&text).unwrap().remove(0);
        assert_eq!(hmm.abc_type, AlphabetType::Amino);
        assert!(hmm.flags & P7H_STATS != 0);
        assert!((hmm.evparam[P7_MMU] - (-6.4582)).abs() < 1e-3);
        assert!((hmm.evparam[P7_FTAU] - (-4.5231)).abs() < 1e-3);
    }

    #[test]
    fn writer_emits_pfam_cutoff_headers() {
        let mut hmm = minimal_hmm("with_cutoffs", AlphabetType::Amino);
        hmm.cutoff[P7_GA1] = 25.0;
        hmm.cutoff[P7_GA2] = 24.5;
        hmm.cutoff[P7_TC1] = 30.0;
        hmm.cutoff[P7_TC2] = 29.5;
        hmm.cutoff[P7_NC1] = -1.0;
        hmm.cutoff[P7_NC2] = -2.0;
        hmm.flags |= P7H_GA | P7H_TC | P7H_NC;

        let mut buf = Vec::new();
        write_hmm(&mut buf, &hmm).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("GA    25.00 24.50\n"), "{text}");
        assert!(text.contains("TC    30.00 29.50\n"), "{text}");
        assert!(text.contains("NC    -1.00 -2.00\n"), "{text}");
    }

    #[test]
    fn writer_pads_stats_local_lines_to_c_field_widths() {
        // C p7_hmmfile.c writes STATS LOCAL lines as
        //   "STATS LOCAL MSV      %8.4f %8.5f\n" (and VITERBI/FORWARD).
        // A mu with magnitude >= 10 produces an 8-char value; C's %8.4f emits
        // no leading pad, so the field stays width 8. Rust must match exactly.
        let mut hmm = minimal_hmm("bigmu", AlphabetType::Amino);
        hmm.flags |= P7H_STATS;
        hmm.evparam[P7_MMU] = -10.8752;
        hmm.evparam[P7_MLAMBDA] = 0.70247;
        hmm.evparam[P7_VMU] = -11.6882;
        hmm.evparam[P7_VLAMBDA] = 0.70247;
        hmm.evparam[P7_FTAU] = -5.2290;
        hmm.evparam[P7_FLAMBDA] = 0.70247;

        let mut buf = Vec::new();
        write_hmm(&mut buf, &hmm).unwrap();
        let text = String::from_utf8(buf).unwrap();

        // Byte-for-byte against C `hmmconvert -a` on a model with mu <= -10.
        assert!(
            text.contains("STATS LOCAL MSV      -10.8752  0.70247\n"),
            "{text}"
        );
        assert!(
            text.contains("STATS LOCAL VITERBI  -11.6882  0.70247\n"),
            "{text}"
        );
        // 7-char mu still right-justifies in width 8 (one leading pad space).
        assert!(
            text.contains("STATS LOCAL FORWARD   -5.2290  0.70247\n"),
            "{text}"
        );
    }

    #[test]
    fn writer_uses_hmmer3f_for_match_mask_annotation() {
        let mut hmm = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .remove(0);
        hmm.flags |= P7H_MMASK;

        let mut buf = Vec::new();
        write_hmm(&mut buf, &hmm).unwrap();
        assert!(String::from_utf8(buf).unwrap().starts_with("HMMER3/f "));
    }

    #[test]
    fn parses_hmmer3e_map_annotations_without_mm_column() {
        let hmms = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/RRM_1.hmm"
        )))
        .unwrap();
        let hmm = &hmms[0];
        assert_eq!(hmm.name, "RRM_1");
        assert_eq!(hmm.map.as_ref().unwrap()[1], 1);
        assert_eq!(hmm.consensus.as_ref().unwrap()[1], b'l');
        assert!(hmm.rf.is_none());
        assert_eq!(hmm.cs.as_ref().unwrap()[1], b'E');
    }

    #[test]
    fn parses_hmmer3a_records_written_by_rust() {
        let hmm = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .remove(0);
        let mut buf = Vec::new();
        write_hmm_with_format(&mut buf, &hmm, HmmAsciiFormat::Hmmer3a).unwrap();
        assert!(String::from_utf8_lossy(&buf).starts_with("HMMER3/a "));

        let hmms = read_hmms(BufReader::new(Cursor::new(buf))).unwrap();
        assert_eq!(hmms.len(), 1);
        assert_eq!(hmms[0].name, "test");
        assert_eq!(hmms[0].m, 20);
        assert_eq!(hmms[0].abc_type, AlphabetType::Amino);
    }

    #[test]
    fn parses_hmmer3f_map_annotations_with_mm_placeholder() {
        let hmms = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/gecco_cluster1_hmms.hmm"
        )))
        .unwrap();
        let hmm = &hmms[0];
        assert_eq!(hmm.name, "Alpha-amylase");
        assert_eq!(hmm.map.as_ref().unwrap()[1], 1);
        assert_eq!(hmm.consensus.as_ref().unwrap()[1], b'G');
        assert!(hmm.rf.is_none());
        assert!(hmm.mm.is_none());
        assert_eq!(hmm.cs.as_ref().unwrap()[1], b'-');
    }

    #[test]
    fn test_read_multiple_hmms() {
        // minipfam has multiple HMMs
        let hmms = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/minipfam.hmm"
        )))
        .unwrap();
        assert!(hmms.len() > 1, "Expected multiple HMMs in minipfam");
    }

    #[test]
    fn rejects_ascii_hmm_database_with_mixed_alphabets() {
        let mut text = Vec::new();
        write_hmm(&mut text, &minimal_hmm("protein", AlphabetType::Amino)).unwrap();
        write_hmm(&mut text, &minimal_hmm("dna", AlphabetType::Dna)).unwrap();

        let err = read_hmms(BufReader::new(Cursor::new(text))).unwrap_err();
        assert!(
            matches!(err, HmmerError::Format(msg) if msg.contains("ASCII HMM file contains mixed alphabets"))
        );
    }

    #[test]
    fn rejects_bad_ga_cutoff_instead_of_defaulting() {
        let err = read_hmms_from_str(
            "HMMER3/f [3.4 | Aug 2023]\nNAME  x\nLENG  1\nALPH  amino\nGA    nope\n",
        )
        .unwrap_err();
        assert!(matches!(err, HmmerError::Format(msg) if msg.contains("Bad GA cutoff")));
    }

    #[test]
    fn rejects_invalid_yes_no_header_values() {
        let original = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/gecco_cluster1_hmms.hmm"
        ))
        .unwrap();
        for key in ["RF", "MM", "CONS", "CS", "MAP"] {
            let yes_needle = format!("{key:<5} yes");
            let no_needle = format!("{key:<5} no");
            let needle = if original.contains(&yes_needle) {
                yes_needle
            } else {
                no_needle
            };
            let mutated = original.replacen(&needle, &format!("{key:<5} maybe"), 1);
            assert_ne!(mutated, original, "test fixture did not contain {needle}");
            let err = read_hmms_from_str(&mutated).unwrap_err();
            assert!(
                matches!(&err, HmmerError::Format(msg) if msg.contains(&format!("Bad {key} value"))),
                "{err}"
            );
        }
    }

    #[test]
    fn writer_preserves_comlog_lines() {
        let mut hmm = minimal_hmm("with_com", AlphabetType::Amino);
        hmm.comlog = Some("first command\nsecond command".to_string());
        let mut buf = Vec::new();
        write_hmm(&mut buf, &hmm).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("COM   [1] first command\n"), "{text}");
        assert!(text.contains("COM   [2] second command\n"), "{text}");
    }

    #[test]
    fn rejects_mixed_ascii_hmmer3_versions() {
        let mut text = Vec::new();
        write_hmm(&mut text, &minimal_hmm("first", AlphabetType::Amino)).unwrap();
        let second = {
            let mut buf = Vec::new();
            write_hmm(&mut buf, &minimal_hmm("second", AlphabetType::Amino)).unwrap();
            String::from_utf8(buf)
                .unwrap()
                .replacen("HMMER3/f", "HMMER3/e", 1)
        };
        text.extend_from_slice(second.as_bytes());

        let err = read_hmms(BufReader::new(Cursor::new(text))).unwrap_err();
        assert!(
            matches!(&err, HmmerError::Format(msg) if msg.contains("mixed HMMER3 format versions")),
            "{err}"
        );
    }

    #[test]
    fn parses_pfam_semicolon_terminated_cutoffs() {
        assert_eq!(parse_cutoff_value("GA", "22;").unwrap(), 22.0);
        assert_eq!(parse_cutoff_value("TC", "20.7;").unwrap(), 20.7);
        assert_eq!(parse_cutoff_value("NC", "29.6").unwrap(), 29.6);

        let text = include_str!("../hmmer/tutorial/fn3.hmm")
            .replace("GA    8.00 7.20", "GA    22;")
            .replace("TC    8.00 7.20", "TC    23.5;")
            .replace("NC    7.90 7.90", "NC    -1.25;");

        let hmms = read_hmms_from_str(&text).unwrap();
        let hmm = &hmms[0];
        assert_eq!(hmm.cutoff[P7_GA1], 22.0);
        assert_eq!(hmm.cutoff[P7_GA2], 22.0);
        assert_eq!(hmm.cutoff[P7_TC1], 23.5);
        assert_eq!(hmm.cutoff[P7_TC2], 23.5);
        assert_eq!(hmm.cutoff[P7_NC1], -1.25);
        assert_eq!(hmm.cutoff[P7_NC2], -1.25);
        assert!(hmm.flags & P7H_GA != 0);
        assert!(hmm.flags & P7H_TC != 0);
        assert!(hmm.flags & P7H_NC != 0);
    }

    #[test]
    fn rejects_truncated_compo_line() {
        let text = "\
HMMER3/f [3.4 | Aug 2023]
NAME  x
LENG  1
ALPH  DNA
HMM          A        C        G        T
            m->m     m->i     m->d     i->m     i->i     d->m     d->d
COMPO   1.38629  1.38629
        1.38629  1.38629  1.38629  1.38629
        0.00000        *        *  0.00000        *        *        *
     1  1.38629  1.38629  1.38629  1.38629
        1.38629  1.38629  1.38629  1.38629
              *        *        *  0.00000        *        *        *
//
";
        let err = read_hmms_from_str(text).unwrap_err();
        assert!(
            matches!(err, HmmerError::Format(msg) if msg.contains("COMPO line has too few fields"))
        );
    }
}
