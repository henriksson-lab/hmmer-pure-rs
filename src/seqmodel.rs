//! Build an HMM from a single sequence using a substitution matrix.
//! Simplified port of seqmodel.c p7_Seqmodel().

use crate::alphabet::{Alphabet, AlphabetType, Dsq};
use crate::bg::Bg;
use crate::calibrate::CalibrationConfig;
use crate::hmm::*;
use crate::util::cmath::c_exp_f64;
use std::io::Read;
use std::path::Path;

const BUILTIN_SCOREMATRIX_SOURCE: &str = include_str!("../hmmer/easel/esl_scorematrix.c");
const MAX_SCORE_MATRIX_FILE_BYTES: usize = 1024 * 1024;
const BUILTIN_MATRIX_NAMES: &[&str] = &[
    "PAM30", "PAM70", "PAM120", "PAM240", "BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80",
    "BLOSUM90",
];
const BUILTIN_NT_MATRIX_NAMES: &[&str] = &["DNA1"];

/// Built-in BLOSUM62 scores for the 20 canonical amino acids
/// (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y). Used as the default and
/// as a fallback if the bundled Easel source layout changes.
const BLOSUM62_20: [[i32; 20]; 20] = [
    [
        4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2,
    ], // A
    [
        0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2,
    ], // C
    [
        -2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3,
    ], // D
    [
        -1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2,
    ], // E
    [
        -2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3,
    ], // F
    [
        0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3,
    ], // G
    [
        -2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2,
    ], // H
    [
        -1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1,
    ], // I
    [
        -1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2,
    ], // K
    [
        -1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1,
    ], // L
    [
        -1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1,
    ], // M
    [
        -2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2,
    ], // N
    [
        -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3,
    ], // P
    [
        -1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1,
    ], // Q
    [
        -1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2,
    ], // R
    [
        1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2,
    ], // S
    [
        0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2,
    ], // T
    [
        0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1,
    ], // V
    [
        -3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2,
    ], // W
    [
        -2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7,
    ], // Y
];

const DNA1_4: [[i32; 4]; 4] = [
    [41, -32, -26, -26],
    [-32, 39, -38, -17],
    [-26, -38, 46, -31],
    [-26, -17, -31, 39],
];

#[derive(Debug, Clone)]
pub struct ScoreMatrix {
    name: String,
    scores: Vec<Vec<i32>>,
    k: usize,
}

impl ScoreMatrix {
    pub fn blosum62() -> Self {
        Self {
            name: "BLOSUM62".to_string(),
            scores: matrix20_to_vec(&BLOSUM62_20),
            k: 20,
        }
    }

    pub fn dna1() -> Self {
        Self {
            name: "DNA1".to_string(),
            scores: matrix4_to_vec(&DNA1_4),
            k: 4,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn builtin(name: &str) -> Result<Self, String> {
        Self::builtin_for_alphabet(name, AlphabetType::Amino)
    }

    pub fn builtin_for_alphabet(name: &str, abc_type: AlphabetType) -> Result<Self, String> {
        match abc_type {
            AlphabetType::Amino => Self::builtin_protein(name),
            AlphabetType::Dna | AlphabetType::Rna => Self::builtin_nucleotide(name),
            AlphabetType::Unknown => Err("unknown alphabet for score matrix".to_string()),
        }
    }

    fn builtin_protein(name: &str) -> Result<Self, String> {
        let canonical = BUILTIN_MATRIX_NAMES
            .iter()
            .copied()
            .find(|candidate| candidate.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                format!(
                    "unknown built-in protein score matrix {name}; supported matrices are {}",
                    BUILTIN_MATRIX_NAMES.join(", ")
                )
            })?;

        let scores = if canonical == "BLOSUM62" {
            matrix20_to_vec(&BLOSUM62_20)
        } else {
            parse_builtin_matrix(canonical).ok_or_else(|| {
                format!("failed to load built-in protein score matrix {canonical}")
            })?
        };

        Ok(Self {
            name: canonical.to_string(),
            scores,
            k: 20,
        })
    }

    fn builtin_nucleotide(name: &str) -> Result<Self, String> {
        let canonical = BUILTIN_NT_MATRIX_NAMES
            .iter()
            .copied()
            .find(|candidate| candidate.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                format!(
                    "unknown built-in nucleotide score matrix {name}; supported matrices are {}",
                    BUILTIN_NT_MATRIX_NAMES.join(", ")
                )
            })?;

        Ok(Self {
            name: canonical.to_string(),
            scores: matrix4_to_vec(&DNA1_4),
            k: 4,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self, String> {
        Self::from_file_for_alphabet(path, &Alphabet::amino())
    }

    pub fn from_file_for_alphabet(path: &Path, abc: &Alphabet) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("failed to read score matrix file {}: {e}", path.display()))?;
        let mut bytes = Vec::new();
        file.by_ref()
            .take((MAX_SCORE_MATRIX_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| format!("failed to read score matrix file {}: {e}", path.display()))?;
        if bytes.len() > MAX_SCORE_MATRIX_FILE_BYTES {
            return Err(format!(
                "failed to read score matrix file {}: file exceeds {} bytes",
                path.display(),
                MAX_SCORE_MATRIX_FILE_BYTES
            ));
        }
        let text = String::from_utf8(bytes).map_err(|e| {
            format!(
                "failed to read score matrix file {}: invalid UTF-8: {e}",
                path.display()
            )
        })?;
        let scores = parse_score_matrix_file(&text, abc)
            .map_err(|e| format!("failed to parse score matrix file {}: {e}", path.display()))?;
        Ok(Self {
            name: path.display().to_string(),
            scores,
            k: abc.k,
        })
    }
}

pub fn is_known_builtin_score_matrix_name(name: &str) -> bool {
    BUILTIN_MATRIX_NAMES
        .iter()
        .chain(BUILTIN_NT_MATRIX_NAMES.iter())
        .any(|candidate| candidate.eq_ignore_ascii_case(name))
}

fn matrix20_to_vec(matrix: &[[i32; 20]; 20]) -> Vec<Vec<i32>> {
    matrix.iter().map(|row| row.to_vec()).collect()
}

fn matrix4_to_vec(matrix: &[[i32; 4]; 4]) -> Vec<Vec<i32>> {
    matrix.iter().map(|row| row.to_vec()).collect()
}

fn parse_builtin_matrix(name: &str) -> Option<Vec<Vec<i32>>> {
    let needle = format!("{{ \"{name}\",");
    let source = &BUILTIN_SCOREMATRIX_SOURCE[BUILTIN_SCOREMATRIX_SOURCE.find(&needle)?..];
    let mut scores = vec![vec![0_i32; 20]; 20];
    let mut row = 0usize;

    for line in source.lines() {
        if row == 20 {
            return Some(scores);
        }
        let trimmed = line.trim_start();
        if !trimmed.starts_with('{') || !line.contains("/*") {
            continue;
        }
        let Some(end) = line.find('}') else {
            continue;
        };
        let fields: Vec<i32> = line[..end]
            .trim_start_matches(|c: char| c == '{' || c.is_whitespace())
            .split(',')
            .filter_map(|field| {
                let field = field.trim();
                (!field.is_empty())
                    .then(|| field.parse::<i32>().ok())
                    .flatten()
            })
            .collect();
        if fields.len() < 20 {
            return None;
        }
        scores[row].copy_from_slice(&fields[..20]);
        row += 1;
    }

    (row == 20).then_some(scores)
}

fn parse_score_matrix_file(text: &str, abc: &Alphabet) -> Result<Vec<Vec<i32>>, String> {
    let mut lines = text.lines().filter_map(matrix_line_tokens);
    let header = lines
        .next()
        .ok_or_else(|| "file appears to be empty".to_string())?;
    if header.is_empty() {
        return Err("header is empty".to_string());
    }

    let mut col_map = Vec::with_capacity(header.len());
    let mut seen = vec![false; abc.k];
    for token in &header {
        if token.len() != 1 {
            return Err(format!(
                "header labels must be single residues; {token} is invalid"
            ));
        }
        let idx = matrix_symbol_index(abc, token.as_bytes()[0])
            .ok_or_else(|| format!("unknown residue {token} in matrix header"))?;
        col_map.push(idx);
        if idx < abc.k {
            seen[idx] = true;
        }
    }
    for (idx, residue) in abc.sym.iter().take(abc.k).enumerate() {
        if !seen[idx] {
            return Err(format!(
                "expected to see a column for residue {}",
                *residue as char
            ));
        }
    }

    let mut scores = vec![vec![0_i32; abc.k]; abc.k];
    let mut filled = vec![false; abc.k];
    for row_number in 0..header.len() {
        let row = lines
            .next()
            .ok_or_else(|| "unexpectedly ran out of matrix rows".to_string())?;
        let mut offset = 0usize;
        let row_idx = if row.len() == header.len() + 1 {
            if row[0].len() != 1 {
                return Err("row labels must be single residues".to_string());
            }
            offset = 1;
            matrix_symbol_index(abc, row[0].as_bytes()[0])
                .ok_or_else(|| format!("unknown residue {} in matrix row", row[0]))?
        } else if row.len() == header.len() {
            col_map[row_number]
        } else {
            return Err("matrix rows must contain one score per header column".to_string());
        };

        if row_idx >= abc.k {
            continue;
        }
        filled[row_idx] = true;
        for (col, &col_idx) in col_map.iter().enumerate() {
            if col_idx >= abc.k {
                continue;
            }
            scores[row_idx][col_idx] = row[col + offset]
                .parse::<i32>()
                .map_err(|_| format!("invalid score {}", row[col + offset]))?;
        }
    }

    if lines.next().is_some() {
        return Err("too many lines in matrix file".to_string());
    }
    for (idx, residue) in abc.sym.iter().take(abc.k).enumerate() {
        if !filled[idx] {
            return Err(format!(
                "expected to see a row for residue {}",
                *residue as char
            ));
        }
    }

    Ok(scores)
}

fn matrix_symbol_index(abc: &Alphabet, residue: u8) -> Option<usize> {
    if residue as usize >= abc.inmap.len() {
        return None;
    }
    let code = abc.digitize_symbol(residue);
    if code == crate::alphabet::DSQ_ILLEGAL
        || code == crate::alphabet::DSQ_IGNORED
        || code == crate::alphabet::DSQ_SENTINEL
    {
        None
    } else {
        Some(code as usize)
    }
}

fn matrix_line_tokens(line: &str) -> Option<Vec<String>> {
    let line = line
        .split_once('#')
        .map_or(line, |(prefix, _)| prefix)
        .trim();
    (!line.is_empty()).then(|| line.split_whitespace().map(str::to_string).collect())
}

/// Reverse-engineer a score matrix into a conditional probability matrix `P(b|a)`
/// given background frequencies `bg_f`.
///
/// Solves for the matrix's natural scale `λ` (so that the implied joint
/// distribution sums to 1), forms `P(a,b) = f(a) f(b) exp(λ · s(a,b))`,
/// then normalises each row to get `P(target=b | query=a)`. Used to seed
/// HMMER's single-sequence query model (`p7_Seqmodel`).
fn score_to_conditional(
    matrix: &ScoreMatrix,
    abc: &Alphabet,
    bg_f: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    let k = abc.k;
    if matrix.k != k {
        return Err(format!(
            "score matrix {} is for alphabet size {}, but model alphabet has size {}",
            matrix.name, matrix.k, k
        ));
    }
    let lambda = solve_lambda(&matrix.scores, bg_f)?;

    let mut joint = vec![vec![0.0_f64; k]; k];
    for a in 0..k {
        for b in 0..k {
            joint[a][b] = (bg_f[a] as f64)
                * (bg_f[b] as f64)
                * c_exp_f64(lambda * matrix.scores[a][b] as f64);
        }
    }

    let mut cond = vec![vec![0.0_f32; k]; abc.kp.max(k)];
    for residue in 0..cond.len() {
        let mut row = vec![0.0_f64; k];
        if residue < k {
            row.copy_from_slice(&joint[residue]);
        } else if residue < abc.degen.len() && abc.ndegen[residue] > 0 {
            for (a, joint_row) in joint.iter().enumerate().take(k) {
                if abc.degen[residue][a] {
                    for (b, cell) in row.iter_mut().enumerate().take(k) {
                        *cell += joint_row[b];
                    }
                }
            }
        }

        let row_sum: f64 = row.iter().sum();
        for b in 0..k {
            cond[residue][b] = if row_sum > 0.0 {
                (row[b] / row_sum) as f32
            } else {
                bg_f[b]
            };
        }
    }
    Ok(cond)
}

/// Residual of the lambda-fixing equation:
/// `sum_{a,b} f(a) f(b) exp(λ s_{ab}) - 1`.
/// Zero when `λ` is the natural scale of the score matrix.
fn lambda_f(scores: &[Vec<i32>], bg_f: &[f32], lambda: f64) -> f64 {
    let mut fx = -1.0_f64;
    for a in 0..scores.len() {
        for b in 0..scores.len() {
            fx += (bg_f[a] as f64) * (bg_f[b] as f64) * c_exp_f64(lambda * scores[a][b] as f64);
        }
    }
    fx
}

/// Bisect the root of [`lambda_f`] to find the score matrix's natural scale `λ`
/// given the background `bg_f`. Returns the `f64` lambda value.
fn solve_lambda(scores: &[Vec<i32>], bg_f: &[f32]) -> Result<f64, String> {
    let max_score = scores
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap_or(0) as f64;
    if max_score <= 0.0 {
        return Err("score matrix has no positive scores".to_string());
    }
    let mut hi = 1.0 / max_score;
    while hi < 50.0 && lambda_f(scores, bg_f, hi) <= 0.0 {
        hi *= 2.0;
    }
    if lambda_f(scores, bg_f, hi) <= 0.0 {
        return Err("failed to bracket lambda root for score matrix".to_string());
    }
    let mut lo = 0.0_f64;
    for _ in 0..80 {
        let mid = (lo + hi) * 0.5;
        if lambda_f(scores, bg_f, mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok((lo + hi) * 0.5)
}

/// Compute both match-state and insert-state occupancies for `hmm`
/// (the local extension of `p7_hmm_CalculateOccupancy` used by
/// `p7_hmm_SetComposition` in C).
fn calculate_occupancy(hmm: &Hmm) -> (Vec<f32>, Vec<f32>) {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    let mut iocc = vec![0.0_f32; hmm.m + 1];

    mocc[0] = 0.0;
    mocc[1] = hmm.t[0][MI] + hmm.t[0][MM];
    for k in 2..=hmm.m {
        let prev = mocc[k - 1];
        let match_or_insert = prev * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI]);
        let delete_entry = (1.0_f64 - prev as f64) * hmm.t[k - 1][DM] as f64;
        mocc[k] = (match_or_insert as f64 + delete_entry) as f32;
    }

    iocc[0] = hmm.t[0][MI] / hmm.t[0][IM];
    for k in 1..=hmm.m {
        iocc[k] = mocc[k] * hmm.t[k][MI] / hmm.t[k][IM];
    }

    (mocc, iocc)
}

/// Set `hmm.compo[]` to the model's expected residue composition
/// (port of `p7_hmm_SetComposition`).
///
/// Weights each match emission by its occupancy and each insert emission by
/// its insert-state occupancy, then normalises the result. Raises `P7H_COMPO`.
fn set_composition(hmm: &mut Hmm) {
    let (mocc, iocc) = calculate_occupancy(hmm);
    for x in 0..hmm.abc_k.min(MAXABET) {
        hmm.compo[x] = hmm.ins[0][x] * iocc[0];
    }
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k.min(MAXABET) {
            hmm.compo[x] += hmm.mat[k][x] * mocc[k] + hmm.ins[k][x] * iocc[k];
        }
    }

    let sum: f32 = hmm.compo[..hmm.abc_k.min(MAXABET)].iter().sum();
    if sum > 0.0 {
        for x in 0..hmm.abc_k.min(MAXABET) {
            hmm.compo[x] /= sum;
        }
    }
    hmm.flags |= P7H_COMPO;
}

/// Build a profile HMM from one query sequence (port of `p7_Seqmodel`).
///
/// Probabilistic Smith/Waterman-style query model: match emissions are derived
/// from a substitution score matrix as conditional distributions
/// `P(b | dsq[k])`, insert emissions are the background, and transitions use
/// `popen` for gap-open (`t_MI`, `t_MD`) and `pextend` for gap-extend
/// (`t_II`, `t_DD`). Node M gets the usual termination tweaks. Sets composition (via
/// `set_composition`) and calibrates E-value statistics by simulation.
pub fn build_single_seq_hmm(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    popen: f32,
    pextend: f32,
) -> Hmm {
    build_single_seq_hmm_with_matrix(
        name,
        dsq,
        seq_len,
        abc,
        bg,
        &ScoreMatrix::blosum62(),
        popen,
        pextend,
    )
    .expect("default BLOSUM62 score matrix should be valid")
}

#[allow(clippy::too_many_arguments)]
pub fn build_single_seq_hmm_with_matrix(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &ScoreMatrix,
    popen: f32,
    pextend: f32,
) -> Result<Hmm, String> {
    build_single_seq_hmm_with_matrix_and_calibration(
        name,
        dsq,
        seq_len,
        abc,
        bg,
        matrix,
        popen,
        pextend,
        42,
        CalibrationConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_single_seq_hmm_with_matrix_and_calibration(
    name: &str,
    dsq: &[Dsq],
    seq_len: usize,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &ScoreMatrix,
    popen: f32,
    pextend: f32,
    calibration_seed: u32,
    calibration_config: CalibrationConfig,
) -> Result<Hmm, String> {
    let k = abc.k;
    let m = seq_len;

    let cond = score_to_conditional(matrix, abc, &bg.f)?;

    let mut hmm = Hmm::new(m, abc.abc_type, k);
    hmm.name = name.to_string();

    // Mirror C p7_Seqmodel (hmmer/src/seqmodel.c:55) exactly: set transitions
    // for every node k in 0..=M with the same formula, then override a
    // subset of node M's transitions at the end. Rust previously hand-wrote
    // node 0 with t[0][MM]=1-popen and node M with all-zeroed I/D
    // transitions, producing ~21-bit score inflation vs C phmmer.
    let mut node = 0usize;
    while node <= m {
        // Match emissions from conditional probability matrix (only for k>0).
        if node > 0 {
            let residue = dsq[node] as usize;
            if residue < cond.len() {
                hmm.mat[node][..k].copy_from_slice(&cond[residue][..k]);
            } else {
                hmm.mat[node][..k].copy_from_slice(&bg.f[..k]);
            }
        }

        // Insert emissions = background, for every node including 0.
        hmm.ins[node][..k].copy_from_slice(&bg.f[..k]);

        hmm.t[node][MM] = 1.0 - 2.0 * popen;
        hmm.t[node][MI] = popen;
        hmm.t[node][MD] = popen;
        hmm.t[node][IM] = 1.0 - pextend;
        hmm.t[node][II] = pextend;
        hmm.t[node][DM] = 1.0 - pextend;
        hmm.t[node][DD] = pextend;
        node += 1;
    }

    // Special handling at node M (C seqmodel.c:85): overrides MM, MD, DM, DD
    // ONLY. MI, IM, II keep their general-formula values from the loop above.
    hmm.t[m][MM] = 1.0 - popen;
    hmm.t[m][MD] = 0.0;
    hmm.t[m][DM] = 1.0;
    hmm.t[m][DD] = 0.0;

    // Insert emissions at node 0
    for x in 0..k {
        hmm.ins[0][x] = bg.f[x];
    }

    set_composition(&mut hmm);

    // Set consensus from sequence. Port of p7_hmm_SetConsensus(hmm, sq)
    // (p7_hmm.c:700-732): with sq given, x = sq->dsq[k]; the symbol is
    // upper-cased when its own match-emission probability mat[k][x] >= mthresh,
    // lower-cased otherwise. mthresh = 0.5 amino, 0.9 DNA/RNA, 0.5 otherwise.
    let mthresh: f32 = match abc.abc_type {
        AlphabetType::Dna | AlphabetType::Rna => 0.9,
        _ => 0.5,
    };
    let mut cons = vec![b' '; m + 2];
    for node in 1..=m {
        let residue = dsq[node] as usize;
        if residue < abc.kp {
            let sym = abc.sym[residue];
            cons[node] = if residue < k && hmm.mat[node][residue] >= mthresh {
                sym.to_ascii_uppercase()
            } else {
                sym.to_ascii_lowercase()
            };
        }
    }
    hmm.consensus = Some(cons);
    hmm.flags |= P7H_CONS;

    // E-value calibration by simulation
    crate::calibrate::calibrate_with_config(
        &mut hmm,
        abc,
        bg,
        calibration_seed,
        calibration_config,
    );

    hmm.nseq = 1;
    hmm.eff_nseq = 1.0;
    hmm.comlog = Some("[HMM created from a query sequence]".to_string());

    Ok(hmm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    /// F3: single-seq consensus is case-thresholded like C's
    /// `p7_hmm_SetConsensus` (`p7_hmm.c:720-721`): a residue whose own
    /// match-emission probability is below `mthresh` (0.5 amino) is
    /// lower-cased, otherwise upper-cased.
    #[test]
    fn single_seq_consensus_case_thresholded() {
        let abc = Alphabet::new(AlphabetType::Amino);
        let bg = Bg::new(&abc);
        // A short protein query. Each residue maps to its canonical symbol;
        // the conditional self-emission prob from BLOSUM62 is < 0.5 for these
        // residues, so all consensus chars must be LOWER case.
        let seq = b"ACDEFGHIKLMNPQRSTVWY";
        let mut dsq = vec![0u8; seq.len() + 2];
        for (i, &c) in seq.iter().enumerate() {
            dsq[i + 1] = abc.digitize_symbol(c);
        }
        let hmm = build_single_seq_hmm("q", &dsq, seq.len(), &abc, &bg, 0.02, 0.4);
        let cons = hmm.consensus.as_ref().expect("consensus set");
        for node in 1..=hmm.m {
            let residue = dsq[node] as usize;
            let p = hmm.mat[node][residue];
            let ch = cons[node];
            if p >= 0.5 {
                assert!(
                    ch.is_ascii_uppercase(),
                    "node {node} p={p} should be UPPER, got {}",
                    ch as char
                );
            } else {
                assert!(
                    ch.is_ascii_lowercase(),
                    "node {node} p={p} should be lower, got {}",
                    ch as char
                );
            }
        }
        // Sanity: at least one residue is below threshold (lower-cased), which
        // is the behaviour the old code got wrong (always upper-cased).
        assert!(
            (1..=hmm.m).any(|k| cons[k].is_ascii_lowercase()),
            "expected at least one lower-cased consensus residue"
        );
    }

    #[test]
    fn score_matrix_reader_rejects_oversized_file_before_full_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("huge.mx");
        std::fs::write(&path, vec![b'A'; MAX_SCORE_MATRIX_FILE_BYTES + 1]).unwrap();

        let err = ScoreMatrix::from_file_for_alphabet(&path, &Alphabet::amino()).unwrap_err();

        assert!(err.contains("exceeds"), "unexpected error: {err}");
    }
}
