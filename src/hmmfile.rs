//! HMM file I/O: reading and writing HMMER3 format HMM files.
//! Direct port of p7_hmmfile.c.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::alphabet::{Alphabet, AlphabetType};
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;

unsafe extern "C" {
    fn logf(x: f32) -> f32;
}

/// Open an HMM save file and read every HMM contained in it.
///
/// Wrapper over [`read_hmms`] that opens `path` first. Returns all HMMs
/// in the file as a `Vec`; errors propagate I/O and format failures.
pub fn read_hmm_file(path: &Path) -> HmmerResult<Vec<Hmm>> {
    let file = std::fs::File::open(path).map_err(|e| HmmerError::Io(e))?;
    let reader = BufReader::new(file);
    read_hmms(reader)
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

    loop {
        match read_one_hmm(&mut lines)? {
            Some(hmm) => hmms.push(hmm),
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
fn read_one_hmm<B: BufRead>(lines: &mut std::io::Lines<B>) -> HmmerResult<Option<Hmm>> {
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

    // Determine format version
    let format_version = if header.starts_with("HMMER3/f") {
        "3f"
    } else if header.starts_with("HMMER3/e") {
        "3e"
    } else if header.starts_with("HMMER3/d") {
        "3d"
    } else if header.starts_with("HMMER3/c") {
        "3c"
    } else if header.starts_with("HMMER3/b") {
        "3b"
    } else if header.starts_with("HMMER3/a") {
        "3a"
    } else {
        "3a" // Default
    };
    if format_version == "3a" {
        return Err(HmmerError::Format(
            "Unsupported legacy HMMER3/a ASCII HMM format".to_string(),
        ));
    }

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
            }
            "MAXL" => {
                max_length = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad MAXL".to_string()))?;
            }
            "ALPH" => {
                abc_type = match value {
                    "amino" => AlphabetType::Amino,
                    "DNA" => AlphabetType::Dna,
                    "RNA" => AlphabetType::Rna,
                    _ => return Err(HmmerError::Format(format!("Unknown alphabet: {}", value))),
                };
            }
            "RF" => rf_flag = value == "yes",
            "MM" => mm_flag = value == "yes" && format_version == "3f",
            "CONS" => cons_flag = value == "yes",
            "CS" => cs_flag = value == "yes",
            "MAP" => map_flag = value == "yes",
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
            }
            "EFFN" => {
                eff_nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad EFFN".to_string()))?;
            }
            "CKSUM" => {
                checksum = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad CKSUM".to_string()))?;
                flags |= P7H_CHKSUM;
            }
            "STATS" => {
                // "LOCAL MSV -6.4582 0.72049"
                let parts: Vec<&str> = value.split_whitespace().collect();
                if parts.len() >= 4 && parts[0] == "LOCAL" {
                    let v1: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    let v2: f32 = parts[3]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    match parts[1] {
                        "MSV" => {
                            evparam[P7_MMU] = v1;
                            evparam[P7_MLAMBDA] = v2;
                            stat_msv = true;
                        }
                        "VITERBI" => {
                            evparam[P7_VMU] = v1;
                            evparam[P7_VLAMBDA] = v2;
                            stat_viterbi = true;
                        }
                        "FORWARD" => {
                            evparam[P7_FTAU] = v1;
                            evparam[P7_FLAMBDA] = v2;
                            stat_forward = true;
                        }
                        _ => {}
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

        let has_cons_column = matches!(format_version, "3e" | "3f");
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
            hmm.mat[node][i] = (-hmm.mat[node][i]).exp();
            hmm.ins[node][i] = (-hmm.ins[node][i]).exp();
        }
        for i in 0..NTRANSITIONS {
            if hmm.t[node][i] != f32::INFINITY {
                hmm.t[node][i] = (-hmm.t[node][i]).exp();
            } else {
                hmm.t[node][i] = 0.0;
            }
        }
    }

    // Convert compo from -ln(prob) to prob
    for i in 0..k.min(MAXABET) {
        if hmm.compo[i] != COMPO_UNSET {
            hmm.compo[i] = (-hmm.compo[i]).exp();
        }
    }

    Ok(Some(hmm))
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
pub fn write_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    let k = hmm.abc_k;
    let write_3f = true;

    if write_3f {
        writeln!(w, "HMMER3/f [3.4 | Aug 2023]").map_err(HmmerError::Io)?;
    } else {
        writeln!(w, "HMMER3/e [3.0 | March 2010]").map_err(HmmerError::Io)?;
    }
    writeln!(w, "NAME  {}", hmm.name).map_err(HmmerError::Io)?;
    if let Some(ref acc) = hmm.acc {
        writeln!(w, "ACC   {}", acc).map_err(HmmerError::Io)?;
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(w, "DESC  {}", desc).map_err(HmmerError::Io)?;
    }
    writeln!(w, "LENG  {}", hmm.m).map_err(HmmerError::Io)?;
    if hmm.max_length >= 0 {
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
    if hmm.nseq >= 0 {
        writeln!(w, "NSEQ  {}", hmm.nseq).map_err(HmmerError::Io)?;
    }
    if hmm.eff_nseq >= 0.0 {
        writeln!(w, "EFFN  {:.6}", hmm.eff_nseq).map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_CHKSUM != 0 {
        writeln!(w, "CKSUM {}", hmm.checksum).map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_STATS != 0 {
        writeln!(
            w,
            "STATS LOCAL MSV       {:.4}  {:.5}",
            hmm.evparam[P7_MMU], hmm.evparam[P7_MLAMBDA]
        )
        .map_err(HmmerError::Io)?;
        writeln!(
            w,
            "STATS LOCAL VITERBI   {:.4}  {:.5}",
            hmm.evparam[P7_VMU], hmm.evparam[P7_VLAMBDA]
        )
        .map_err(HmmerError::Io)?;
        writeln!(
            w,
            "STATS LOCAL FORWARD   {:.4}  {:.5}",
            hmm.evparam[P7_FTAU], hmm.evparam[P7_FLAMBDA]
        )
        .map_err(HmmerError::Io)?;
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
        write!(w, "  COMPO  ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.compo[i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    // Node 0 insert emissions
    write!(w, "         ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, " {}", fmt_prob(hmm.ins[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Node 0 transitions
    write!(w, "         ").map_err(HmmerError::Io)?;
    for i in 0..NTRANSITIONS {
        write!(w, " {}", fmt_prob(hmm.t[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Nodes 1..M
    for node in 1..=hmm.m {
        // Match emission line
        write!(w, " {:>6}", node).map_err(HmmerError::Io)?;
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
        write!(w, " {} {}", cons_ch, rf_ch).map_err(HmmerError::Io)?;
        if write_3f {
            write!(w, " {}", mm_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", cs_ch).map_err(HmmerError::Io)?;
        writeln!(w).map_err(HmmerError::Io)?;

        // Insert emission line
        write!(w, "         ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.ins[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Transition line
        write!(w, "         ").map_err(HmmerError::Io)?;
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
fn fmt_prob(p: f32) -> String {
    if p <= 0.0 {
        "      *".to_string()
    } else if p == 1.0 {
        format!("{:7.5}", 0.0)
    } else {
        // HMMER's C writer uses logf(), not double-precision log().
        format!("{:7.5}", -unsafe { logf(p) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};
    use std::path::Path;

    fn read_hmms_from_str(s: &str) -> HmmerResult<Vec<Hmm>> {
        read_hmms(BufReader::new(Cursor::new(s.as_bytes())))
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
    fn rejects_bad_ga_cutoff_instead_of_defaulting() {
        let err = read_hmms_from_str(
            "HMMER3/f [3.4 | Aug 2023]\nNAME  x\nLENG  1\nALPH  amino\nGA    nope\n",
        )
        .unwrap_err();
        assert!(matches!(err, HmmerError::Format(msg) if msg.contains("Bad GA cutoff")));
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
