//! HMM file I/O: reading and writing HMMER3 format HMM files.
//! Direct port of p7_hmmfile.c.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::alphabet::AlphabetType;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;

/// Read all HMMs from an HMM file.
pub fn read_hmm_file(path: &Path) -> HmmerResult<Vec<Hmm>> {
    let file = std::fs::File::open(path).map_err(|e| HmmerError::Io(e))?;
    let reader = BufReader::new(file);
    read_hmms(reader)
}

/// Read HMMs from a reader.
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

/// Read a single HMM from a line iterator. Returns None at EOF.
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

    // Read header lines until "HMM" line
    loop {
        let line = lines
            .next()
            .ok_or_else(|| HmmerError::Format("Unexpected EOF in HMM header".to_string()))?
            .map_err(HmmerError::Io)?;
        let trimmed = line.trim();

        if trimmed.starts_with("HMM ") || trimmed == "HMM" {
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
                        }
                        "VITERBI" => {
                            evparam[P7_VMU] = v1;
                            evparam[P7_VLAMBDA] = v2;
                        }
                        "FORWARD" => {
                            evparam[P7_FTAU] = v1;
                            evparam[P7_FLAMBDA] = v2;
                        }
                        _ => {}
                    }
                    flags |= P7H_STATS;
                }
            }
            "GA" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_GA1] = v.parse().unwrap_or(CUTOFF_UNSET);
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_GA2] = v.parse().unwrap_or(cutoff[P7_GA1]);
                } else {
                    cutoff[P7_GA2] = cutoff[P7_GA1];
                }
                flags |= P7H_GA;
            }
            "TC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_TC1] = v.parse().unwrap_or(CUTOFF_UNSET);
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_TC2] = v.parse().unwrap_or(cutoff[P7_TC1]);
                } else {
                    cutoff[P7_TC2] = cutoff[P7_TC1];
                }
                flags |= P7H_TC;
            }
            "NC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_NC1] = v.parse().unwrap_or(CUTOFF_UNSET);
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_NC2] = v.parse().unwrap_or(cutoff[P7_NC1]);
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
        for i in 0..k.min(MAXABET) {
            if let Some(&val) = parts.get(i + 1) {
                hmm.compo[i] = parse_hmm_value(val);
            }
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
        for i in 0..k {
            if let Some(&val) = parts.get(i + 1) {
                hmm.mat[node][i] = parse_hmm_value(val);
            }
        }

        // Parse annotations after the K values
        let annot_start = k + 1;
        if map_flag {
            if let Some(&val) = parts.get(annot_start) {
                if let Some(map) = &mut hmm.map {
                    map[node] = val.parse().unwrap_or(0);
                }
            }
        }
        if cons_flag {
            if let Some(&val) = parts.get(annot_start + 1) {
                if let Some(cons) = &mut hmm.consensus {
                    cons[node] = val.as_bytes().first().copied().unwrap_or(b'-');
                }
            }
        }
        if rf_flag {
            if let Some(&val) = parts.get(annot_start + 2) {
                if let Some(rf) = &mut hmm.rf {
                    rf[node] = val.as_bytes().first().copied().unwrap_or(b'-');
                }
            }
        }
        // CS is always the LAST token on a match-state line in HMMER3 format
        // (placeholders `-` are emitted even for disabled RF/MM). mm lives just
        // before cs if both are enabled, otherwise at the last slot.
        if cs_flag {
            if let Some(&val) = parts.last() {
                if let Some(cs) = &mut hmm.cs {
                    cs[node] = val.as_bytes().first().copied().unwrap_or(b'-');
                }
            }
        }
        if mm_flag {
            // mm sits at the position just before cs (or last if no cs).
            let mm_idx = if cs_flag { parts.len().saturating_sub(2) } else { parts.len().saturating_sub(1) };
            if let Some(&val) = parts.get(mm_idx) {
                if let Some(mm) = &mut hmm.mm {
                    mm[node] = val.as_bytes().first().copied().unwrap_or(b'-');
                }
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
    loop {
        match lines.next() {
            None => break,
            Some(Err(e)) => return Err(HmmerError::Io(e)),
            Some(Ok(line)) => {
                if line.trim() == "//" {
                    break;
                }
            }
        }
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

/// Parse an HMM value: either a float or "*" (which means 0 probability, stored as infinity).
fn parse_hmm_value(s: &str) -> f32 {
    if s == "*" {
        f32::INFINITY
    } else {
        s.parse().unwrap_or(0.0)
    }
}

/// Parse an emission line (K whitespace-separated values).
fn parse_emission_line(line: &str, k: usize, values: &mut [f32]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    for i in 0..k {
        if let Some(&val) = parts.get(i) {
            values[i] = parse_hmm_value(val);
        }
    }
    Ok(())
}

/// Parse a transition line (7 whitespace-separated values).
fn parse_transition_line(line: &str, values: &mut [f32; NTRANSITIONS]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    for i in 0..NTRANSITIONS {
        if let Some(&val) = parts.get(i) {
            values[i] = parse_hmm_value(val);
        }
    }
    Ok(())
}

/// Write an HMM in HMMER3/e ASCII format.
pub fn write_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    let k = hmm.abc_k;

    writeln!(w, "HMMER3/e [3.0 | March 2010]").map_err(HmmerError::Io)?;
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
        writeln!(w, "EFFN  {}", hmm.eff_nseq).map_err(HmmerError::Io)?;
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
        // Annotations
        if let Some(ref map) = hmm.map {
            write!(w, " {:>6}", map[node]).map_err(HmmerError::Io)?;
        }
        if let Some(ref cons) = hmm.consensus {
            write!(w, " {}", cons[node] as char).map_err(HmmerError::Io)?;
        }
        if let Some(ref rf) = hmm.rf {
            write!(w, " {}", rf[node] as char).map_err(HmmerError::Io)?;
        }
        if let Some(ref cs) = hmm.cs {
            write!(w, " {}", cs[node] as char).map_err(HmmerError::Io)?;
        }
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

/// Format a probability as -ln(p) for HMM file output.
fn fmt_prob(p: f32) -> String {
    if p <= 0.0 {
        "      *".to_string()
    } else {
        format!("{:7.5}", -p.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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
    fn test_read_multiple_hmms() {
        // minipfam has multiple HMMs
        let hmms = read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/minipfam.hmm"
        )))
        .unwrap();
        assert!(hmms.len() > 1, "Expected multiple HMMs in minipfam");
    }
}
