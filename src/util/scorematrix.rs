//! Substitution score matrices.
//! Port of Easel's esl_scorematrix.c (the slice HMMER's builder uses).
//!
//! Covers `ESL_SCOREMATRIX` allocation, the built-in preloads, matrix-file
//! reading, and the reverse-engineering of scores into conditional residue
//! probabilities that `p7_builder_{Load,Set}ScoreSystem()` needs
//! (p7_builder.c:196, :283).
//!
//! The preload tables are scraped at first use from the vendored upstream
//! source in `src/data/esl_scorematrix.c`, which is byte-identical to Easel
//! 0.49 (`EddyRivasLab/easel@07ca83b`, the release paired with HMMER 3.4 at
//! `EddyRivasLab/hmmer@9acd8b6758a0`) and ships in the crate. Reading the
//! numbers straight from upstream is more faithful than transcribing them.
//!
//! Deviation from C worth knowing: `ESL_SCOREMATRIX` holds `abc_r`, a borrowed
//! pointer to its alphabet. Carrying that as a Rust lifetime would infect every
//! caller, so this port stores only `k`/`kp`/`abc_type` and takes `&Alphabet`
//! on the methods that need `sym`/`degen` — exactly where C dereferences
//! `S->abc_r`.

use crate::alphabet::{Alphabet, AlphabetType, DSQ_IGNORED, DSQ_ILLEGAL, DSQ_SENTINEL};
use crate::util::cmath::c_exp_f64;
use crate::util::rootfinder::newton_raphson;
use std::sync::OnceLock;

/// The vendored Easel source the built-in tables are read from.
const BUILTIN_SCOREMATRIX_SOURCE: &str = include_str!("../data/esl_scorematrix.c");

/// `eslAADIM` (esl_scorematrix.c:726): the amino preloads are 29x29, in Easel
/// digital order `A C D E F G H I K L M N P Q R S T V W Y - B J Z O U X * ~`.
const ESL_AADIM: usize = 29;
/// `eslNTDIM` (esl_scorematrix.c:759): nucleotide preloads are 18x18, in order
/// `A C G T - R Y M K S W H B V D N * ~`.
const ESL_NTDIM: usize = 18;

/// Names in `ESL_SCOREMATRIX_AA_PRELOADS` (esl_scorematrix.c:431), in order.
const AA_PRELOAD_NAMES: &[&str] = &[
    "PAM30", "PAM70", "PAM120", "PAM240", "BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80",
    "BLOSUM90",
];
/// Names in `ESL_SCOREMATRIX_NT_PRELOADS` (esl_scorematrix.c:762).
const NT_PRELOAD_NAMES: &[&str] = &["DNA1"];

/// Residue order C writes into `S->outorder` for the amino preloads
/// (esl_scorematrix.c:832).
const AA_PRELOAD_OUTORDER: &str = "ARNDCQEGHILKMFPSTWYVBZX*";
/// Residue order for the nucleotide preloads (esl_scorematrix.c:848).
const NT_PRELOAD_OUTORDER: &str = "ACGTRYMKSWHBVDN";

/// Mirrors the status codes `esl_scorematrix_Set()` distinguishes, because
/// `p7_builder_LoadScoreSystem()` (p7_builder.c:213-215) reports them
/// differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreMatrixError {
    /// C `eslENOTFOUND`: no built-in by that name for this alphabet.
    NotFound(String),
    /// C `eslEFORMAT` / `eslEINVAL`, carrying C's `errbuf` text.
    Invalid(String),
}

impl std::fmt::Display for ScoreMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScoreMatrixError::NotFound(m) | ScoreMatrixError::Invalid(m) => f.write_str(m),
        }
    }
}

/// Port of `ESL_SCOREMATRIX` (esl_scorematrix.h:23-38).
#[derive(Debug, Clone)]
pub struct ScoreMatrix {
    /// `s[i][j]`, the score of aligning residues i,j; both range `0..Kp-1`.
    pub s: Vec<Vec<i32>>,
    /// `S->K`, size of the base alphabet.
    pub k: usize,
    /// `S->Kp`, full size of `s[][]` including degeneracies.
    pub kp: usize,
    /// `S->isval[0..Kp-1]`: which residues have valid scores.
    pub isval: Vec<bool>,
    /// `S->nc`, number of residues with scores (inclusive of `*` if present).
    pub nc: usize,
    /// `S->outorder`, the residue order of the col/row labels.
    pub outorder: String,
    /// `S->name`, optional.
    pub name: Option<String>,
    /// `S->path`, optional.
    pub path: Option<String>,
    /// Stands in for `S->abc_r->type`.
    pub abc_type: AlphabetType,
}

impl ScoreMatrix {
    /// Port of `esl_scorematrix_Create()` (esl_scorematrix.c:49). All scores
    /// zeroed, `isval` all FALSE, `outorder` empty.
    pub fn create(abc: &Alphabet) -> Self {
        ScoreMatrix {
            s: vec![vec![0_i32; abc.kp]; abc.kp],
            k: abc.k,
            kp: abc.kp,
            isval: vec![false; abc.kp],
            nc: 0,
            outorder: String::new(),
            name: None,
            path: None,
            abc_type: abc.abc_type,
        }
    }

    /// Port of `esl_scorematrix_Set()` (esl_scorematrix.c:814). Sets this
    /// matrix to the built-in named `name`, which must exist for the matrix's
    /// alphabet. Amino matrices are PAM30/70/120/240 and BLOSUM45/50/62/80/90;
    /// the only nucleotide built-in is DNA1.
    pub fn set(&mut self, name: &str, abc: &Alphabet) -> Result<(), ScoreMatrixError> {
        let (table, outorder) = match abc.abc_type {
            AlphabetType::Amino => {
                let which = AA_PRELOAD_NAMES
                    .iter()
                    .position(|n| *n == name)
                    .ok_or_else(|| ScoreMatrixError::NotFound(name.to_string()))?;
                (&aa_preloads()[which], AA_PRELOAD_OUTORDER)
            }
            AlphabetType::Dna | AlphabetType::Rna => {
                let which = NT_PRELOAD_NAMES
                    .iter()
                    .position(|n| *n == name)
                    .ok_or_else(|| ScoreMatrixError::NotFound(name.to_string()))?;
                (&nt_preloads()[which], NT_PRELOAD_OUTORDER)
            }
            // C: "no DNA matrices are built in yet!" — eslENOTFOUND for any
            // other alphabet type (esl_scorematrix.c:856).
            AlphabetType::Unknown => return Err(ScoreMatrixError::NotFound(name.to_string())),
        };

        self.outorder = outorder.to_string();

        // Transfer scores from static built-in storage (esl_scorematrix.c:834-836).
        for x in 0..self.kp {
            for y in 0..self.kp {
                self.s[x][y] = table[x][y];
            }
        }

        // Use <outorder> to set <isval[x]> (esl_scorematrix.c:859-863).
        self.nc = self.outorder.len();
        self.isval.iter_mut().for_each(|v| *v = false);
        for ch in self.outorder.clone().bytes() {
            let x = abc.digitize_symbol(ch);
            if (x as usize) < self.kp {
                self.isval[x as usize] = true;
            }
        }

        self.name = Some(name.to_string());
        Ok(())
    }

    /// Port of `esl_scorematrix_Read()` (esl_scorematrix.c:1044). `text` is the
    /// whole matrix file; `filename` supplies `S->path`/`S->name` the way C
    /// derives them from `efp->filename`.
    ///
    /// Comment character is `#` (esl_scorematrix.c:1060), blank lines are
    /// skipped — that is `ESL_FILEPARSER` behaviour, reproduced by
    /// [`matrix_line_tokens`].
    pub fn read(
        text: &str,
        filename: Option<&str>,
        abc: &Alphabet,
    ) -> Result<Self, ScoreMatrixError> {
        let mut s = ScoreMatrix::create(abc);
        let mut lines = text.lines().filter_map(matrix_line_tokens);

        let header = lines
            .next()
            .ok_or_else(|| ScoreMatrixError::Invalid("file appears to be empty".to_string()))?;

        // Read the label characters into outorder[0..nc-1].
        for tok in &header {
            if s.nc >= abc.kp {
                return Err(ScoreMatrixError::Invalid(
                    "Header contains more residues than expected for alphabet".to_string(),
                ));
            }
            if tok.len() != 1 {
                return Err(ScoreMatrixError::Invalid(format!(
                    "Header can only contain single-char labels; {tok} is invalid"
                )));
            }
            s.outorder.push_str(tok);
            s.nc += 1;
        }

        // Verify the labels against the alphabet; set isval[] and map[].
        let outorder: Vec<u8> = s.outorder.bytes().collect();
        let mut map = vec![0usize; s.nc];
        for (c, &label) in outorder.iter().enumerate() {
            let x = abc.digitize_symbol(label);
            if x == DSQ_ILLEGAL || x == DSQ_IGNORED || x == DSQ_SENTINEL {
                return Err(ScoreMatrixError::Invalid(format!(
                    "Don't know how to deal with residue {} in matrix file",
                    label as char
                )));
            }
            map[c] = x as usize;
            s.isval[x as usize] = true;
        }
        for x in 0..abc.k {
            if !s.isval[x] {
                return Err(ScoreMatrixError::Invalid(format!(
                    "Expected to see a column for residue {}",
                    abc.sym[x] as char
                )));
            }
        }

        // Read nc rows. Each row has nc scores, optionally led by a row label
        // that must equal outorder[row] — C's `if (col == 0 && *tok ==
        // S->outorder[row]) { col--; continue; }` (esl_scorematrix.c:1113).
        for row in 0..s.nc {
            let fields = lines.next().ok_or_else(|| {
                ScoreMatrixError::Invalid("Unexpectedly ran out of lines in file".to_string())
            })?;
            let mut it = fields.iter();
            let mut col = 0usize;
            while col < s.nc {
                let tok = it.next().ok_or_else(|| {
                    ScoreMatrixError::Invalid("Unexpectedly ran out of fields on line".to_string())
                })?;
                if col == 0 && tok.as_bytes().first() == Some(&outorder[row]) {
                    continue; // skip the leading row label
                }
                // C uses atoi(), which yields 0 for unparseable text rather
                // than failing; keep that behaviour.
                s.s[map[row]][map[col]] = c_atoi(tok);
                col += 1;
            }
            if it.next().is_some() {
                return Err(ScoreMatrixError::Invalid(
                    "Too many fields on line".to_string(),
                ));
            }
        }
        if lines.next().is_some() {
            return Err(ScoreMatrixError::Invalid(
                "Too many lines in file. (Make sure it's square & symmetric. E.g. use NUC.4.4 not NUC.4.2)"
                    .to_string(),
            ));
        }

        if let Some(path) = filename {
            s.path = Some(path.to_string());
            s.name = Some(file_tail(path));
        }
        Ok(s)
    }

    /// Convenience: `esl_scorematrix_Create()` + `esl_scorematrix_Set()`, the
    /// pairing every caller of the built-ins actually wants.
    pub fn builtin_for_alphabet(name: &str, abc: &Alphabet) -> Result<Self, ScoreMatrixError> {
        let mut s = ScoreMatrix::create(abc);
        s.set(name, abc)?;
        Ok(s)
    }

    /// Convenience: read a matrix file for `abc`, as
    /// `p7_builder_SetScoreSystem()` does for `--mxfile` (p7_builder.c:307-311).
    pub fn from_file_for_alphabet(
        path: &std::path::Path,
        abc: &Alphabet,
    ) -> Result<Self, ScoreMatrixError> {
        let text = read_matrix_file(path)?;
        ScoreMatrix::read(&text, path.to_str(), abc)
    }

    /// The default built-in for `abc`, i.e. C's
    /// `abc->type == eslAMINO ? "BLOSUM62" : "DNA1"` (p7_builder.c:296-301).
    pub fn default_builtin_name(abc_type: AlphabetType) -> &'static str {
        match abc_type {
            AlphabetType::Amino => "BLOSUM62",
            _ => "DNA1",
        }
    }

    /// BLOSUM62 over the amino alphabet.
    pub fn blosum62() -> Self {
        let abc = Alphabet::new(AlphabetType::Amino);
        ScoreMatrix::builtin_for_alphabet("BLOSUM62", &abc).expect("BLOSUM62 is a built-in")
    }

    /// DNA1 over the DNA alphabet.
    pub fn dna1() -> Self {
        let abc = Alphabet::new(AlphabetType::Dna);
        ScoreMatrix::builtin_for_alphabet("DNA1", &abc).expect("DNA1 is a built-in")
    }

    /// `S->name`, or the empty string if unset.
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("")
    }

    /// Port of `esl_scorematrix_Max()` (esl_scorematrix.c:201). Note it scans
    /// only the canonical `0..K-1` block, not all of `Kp` — the lambda bracket
    /// in [`Self::probify_given_bg`] depends on that.
    pub fn max(&self) -> i32 {
        let mut max = self.s[0][0];
        for i in 0..self.k {
            for j in 0..self.k {
                if self.s[i][j] > max {
                    max = self.s[i][j];
                }
            }
        }
        max
    }

    /// Port of `esl_scorematrix_Min()` (esl_scorematrix.c:218).
    pub fn min(&self) -> i32 {
        let mut min = self.s[0][0];
        for i in 0..self.k {
            for j in 0..self.k {
                if self.s[i][j] < min {
                    min = self.s[i][j];
                }
            }
        }
        min
    }

    /// `lambda_fdf()` (esl_scorematrix.c:1519): returns
    /// `(sum_ij f_i f_j e^{lambda s_ij} - 1, sum_ij f_i f_j s_ij e^{lambda s_ij})`.
    fn lambda_fdf(&self, fi: &[f64], fj: &[f64], lambda: f64) -> (f64, f64) {
        let mut fx = 0.0_f64;
        let mut dfx = 0.0_f64;
        for i in 0..self.k {
            for j in 0..self.k {
                let tmp = fi[i] * fj[j] * c_exp_f64(lambda * self.s[i][j] as f64);
                fx += tmp;
                dfx += tmp * self.s[i][j] as f64;
            }
        }
        fx -= 1.0;
        (fx, dfx)
    }

    /// Port of `esl_scorematrix_ProbifyGivenBG()` (esl_scorematrix.c:1258).
    ///
    /// Solves for lambda by bracketing upward from `1/max()` (doubling while
    /// below 50) until `f(lambda) > 0`, then Newton/Raphson. Returns
    /// `(lambda, P)` where `P` is the `Kp x Kp` joint probability matrix, with
    /// degenerate entries filled in as marginals by [`set_degenerate_probs`].
    pub fn probify_given_bg(
        &self,
        fi: &[f64],
        fj: &[f64],
        abc: &Alphabet,
    ) -> Result<(f64, Vec<Vec<f64>>), ScoreMatrixError> {
        // Bracket the root. It's important that we come at the root from the
        // far side, where f(lambda) is positive; else we may identify the root
        // we don't want at lambda=0. (esl_scorematrix.c:1279-1283)
        let mut fx = -1.0_f64;
        let mut lambda_guess = 1.0 / self.max() as f64;
        while lambda_guess < 50.0 {
            fx = self.lambda_fdf(fi, fj, lambda_guess).0;
            if fx > 0.0 {
                break;
            }
            lambda_guess *= 2.0;
        }
        if fx <= 0.0 {
            return Err(ScoreMatrixError::Invalid(
                "Failed to bracket root for solving lambda".to_string(),
            ));
        }

        let lambda = newton_raphson(|x| self.lambda_fdf(fi, fj, x), lambda_guess)
            .map_err(ScoreMatrixError::Invalid)?;

        let mut p = vec![vec![0.0_f64; self.kp]; self.kp];
        for i in 0..self.k {
            for j in 0..self.k {
                p[i][j] = fi[i] * fj[j] * c_exp_f64(lambda * self.s[i][j] as f64);
            }
        }
        set_degenerate_probs(abc, &mut p, None, None);

        Ok((lambda, p))
    }
}

/// Port of `set_degenerate_probs()` (esl_scorematrix.c:1332).
///
/// Given canonical probabilities in `p`, fills the degenerate rows/columns with
/// marginals summed over the degeneracy, and zeroes everything involving the
/// gap (`K`), nonresidue (`Kp-2`) and missing-data (`Kp-1`) codes. By
/// construction the fully degenerate code `Kp-3` ends up at 1.0.
pub fn set_degenerate_probs(
    abc: &Alphabet,
    p: &mut [Vec<f64>],
    fi: Option<&mut [f64]>,
    fj: Option<&mut [f64]>,
) {
    let k = abc.k;
    let kp = abc.kp;

    // [i][all]: canonical i, degenerate j.
    for i in 0..k {
        p[i][k] = 0.0;
        for jp in (k + 1)..(kp - 2) {
            p[i][jp] = 0.0;
            for j in 0..k {
                if abc.degen[jp][j] {
                    p[i][jp] += p[i][j];
                }
            }
        }
        p[i][kp - 2] = 0.0;
        p[i][kp - 1] = 0.0;
    }

    // Gap row: all 0.0 by convention.
    for v in p[k].iter_mut().take(kp) {
        *v = 0.0;
    }

    // [ip][all]: degenerate i.
    for ip in (k + 1)..(kp - 2) {
        for j in 0..k {
            let mut sum = 0.0;
            for i in 0..k {
                if abc.degen[ip][i] {
                    sum += p[i][j];
                }
            }
            p[ip][j] = sum;
        }
        p[ip][k] = 0.0;

        for jp in (k + 1)..(kp - 2) {
            let mut sum = 0.0;
            for j in 0..k {
                if abc.degen[jp][j] {
                    sum += p[ip][j];
                }
            }
            p[ip][jp] = sum;
        }
        p[ip][kp - 2] = 0.0;
        p[ip][kp - 1] = 0.0;
    }

    // Nonresidue `*` row and missing-data `~` row: all 0.0.
    for v in p[kp - 2].iter_mut().take(kp) {
        *v = 0.0;
    }
    for v in p[kp - 1].iter_mut().take(kp) {
        *v = 0.0;
    }

    if let Some(fi) = fi {
        fi[k] = 0.0;
        for ip in (k + 1)..(kp - 2) {
            fi[ip] = p[ip][kp - 3];
        }
        fi[kp - 2] = 0.0;
        fi[kp - 1] = 0.0;
    }
    if let Some(fj) = fj {
        fj[k] = 0.0;
        for jp in (k + 1)..(kp - 2) {
            fj[jp] = p[kp - 3][jp];
        }
        fj[kp - 2] = 0.0;
        fj[kp - 1] = 0.0;
    }
}

/// Port of `esl_scorematrix_JointToConditionalOnQuery()` (esl_scorematrix.c:369).
///
/// Converts joint `P(a,b)` in place to conditional `P(b | a)`, where `a` is the
/// query residue and `b` the target. `P(a) = P(a,X)`, the value at `[a][Kp-3]`.
pub fn joint_to_conditional_on_query(abc: &Alphabet, p: &mut [Vec<f64>]) {
    let kp = abc.kp;
    for a in 0..(kp - 2) {
        let pa = p[a][kp - 3];
        for b in 0..(kp - 2) {
            p[a][b] = if pa == 0.0 { 0.0 } else { p[a][b] / pa };
        }
    }
}

/// C `atoi()`: parse a leading optional-sign integer, yielding 0 on anything
/// unparseable rather than erroring.
fn c_atoi(tok: &str) -> i32 {
    let bytes = tok.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return 0;
    }
    tok[start..i].parse::<i32>().unwrap_or(0)
}

/// `esl_FileTail(path, FALSE, &name)`: basename, suffix retained.
fn file_tail(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// `ESL_FILEPARSER` line handling with comment char `#`: strip comments, skip
/// blank lines, split on whitespace.
fn matrix_line_tokens(line: &str) -> Option<Vec<String>> {
    let line = line
        .split_once('#')
        .map_or(line, |(prefix, _)| prefix)
        .trim();
    (!line.is_empty()).then(|| line.split_whitespace().map(str::to_string).collect())
}

fn aa_preloads() -> &'static Vec<Vec<Vec<i32>>> {
    static TABLES: OnceLock<Vec<Vec<Vec<i32>>>> = OnceLock::new();
    TABLES.get_or_init(|| {
        AA_PRELOAD_NAMES
            .iter()
            .map(|name| {
                parse_builtin_matrix(name, ESL_AADIM)
                    .unwrap_or_else(|| panic!("bundled Easel source is missing preload {name}"))
            })
            .collect()
    })
}

fn nt_preloads() -> &'static Vec<Vec<Vec<i32>>> {
    static TABLES: OnceLock<Vec<Vec<Vec<i32>>>> = OnceLock::new();
    TABLES.get_or_init(|| {
        NT_PRELOAD_NAMES
            .iter()
            .map(|name| {
                parse_builtin_matrix(name, ESL_NTDIM)
                    .unwrap_or_else(|| panic!("bundled Easel source is missing preload {name}"))
            })
            .collect()
    })
}

/// Scrape one `dim x dim` preload table out of the vendored Easel source.
///
/// The tables are laid out as `{ "NAME", {` followed by a `/* A C ... */`
/// column-label comment and then `dim` rows, each of the form
/// `{  41, -32, ...,   0, }, /*A*/`.
fn parse_builtin_matrix(name: &str, dim: usize) -> Option<Vec<Vec<i32>>> {
    let needle = format!("{{ \"{name}\",");
    let source = &BUILTIN_SCOREMATRIX_SOURCE[BUILTIN_SCOREMATRIX_SOURCE.find(&needle)?..];
    let mut scores = vec![vec![0_i32; dim]; dim];
    let mut row = 0usize;

    for line in source.lines() {
        if row == dim {
            return Some(scores);
        }
        let trimmed = line.trim_start();
        if !trimmed.starts_with('{') || !line.contains("/*") {
            continue;
        }
        let end = line.find('}')?;
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
        if fields.len() < dim {
            return None;
        }
        scores[row].copy_from_slice(&fields[..dim]);
        row += 1;
    }

    (row == dim).then_some(scores)
}

/// Guard against reading an unreasonably large "matrix file".
const MAX_SCORE_MATRIX_FILE_BYTES: usize = 1024 * 1024;

fn read_matrix_file(path: &std::path::Path) -> Result<String, ScoreMatrixError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|_| {
        ScoreMatrixError::Invalid(format!(
            "Failed to find or open matrix file {}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take((MAX_SCORE_MATRIX_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| {
            ScoreMatrixError::Invalid(format!(
                "Failed to read matrix from {}: {e}",
                path.display()
            ))
        })?;
    if bytes.len() > MAX_SCORE_MATRIX_FILE_BYTES {
        return Err(ScoreMatrixError::Invalid(format!(
            "Failed to read matrix from {}: file exceeds {MAX_SCORE_MATRIX_FILE_BYTES} bytes",
            path.display()
        )));
    }
    String::from_utf8(bytes).map_err(|e| {
        ScoreMatrixError::Invalid(format!(
            "Failed to read matrix from {}: invalid UTF-8: {e}",
            path.display()
        ))
    })
}

/// True if `name` is one of Easel's built-in matrices for any alphabet.
/// Used for CLI validation of `--mx`.
pub fn is_known_builtin_score_matrix_name(name: &str) -> bool {
    AA_PRELOAD_NAMES
        .iter()
        .chain(NT_PRELOAD_NAMES.iter())
        .any(|candidate| *candidate == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bg::Bg;

    /// Reference values produced by the C library itself, at
    /// `EddyRivasLab/hmmer@9acd8b6758a0` + `EddyRivasLab/easel@07ca83b`, via
    /// `esl_scorematrix_Set` -> `esl_scorematrix_ProbifyGivenBG` ->
    /// `esl_scorematrix_JointToConditionalOnQuery` with `p7_bg_Create()`
    /// frequencies. Regenerate with `examples/dump_scorematrix.rs` and the
    /// matching C driver; every value below is required to match C bit-for-bit,
    /// not approximately.
    fn conditionals(abc_type: AlphabetType, mxname: &str) -> (f64, Vec<Vec<f64>>, ScoreMatrix) {
        let abc = Alphabet::new(abc_type);
        let bg = Bg::new(&abc);
        let mut s = ScoreMatrix::create(&abc);
        s.set(mxname, &abc).expect("built-in should exist");
        let f: Vec<f64> = bg.f[..abc.k].iter().map(|&x| x as f64).collect();
        let (lambda, mut q) = s.probify_given_bg(&f, &f, &abc).expect("probify");
        joint_to_conditional_on_query(&abc, &mut q);
        (lambda, q, s)
    }

    #[test]
    fn blosum62_matches_c_bitwise() {
        let (lambda, q, s) = conditionals(AlphabetType::Amino, "BLOSUM62");
        assert_eq!(s.k, 20);
        assert_eq!(s.kp, 29);
        assert_eq!(s.max(), 11);
        assert_eq!(s.min(), -4);
        assert_eq!(s.nc, 24);
        assert_eq!(s.outorder, "ARNDCQEGHILKMFPSTWYVBZX*");
        assert_eq!(lambda, 3.17951304112292610e-01);
        // Row 0 (A), first three conditionals P(b|A).
        assert_eq!(q[0][0], 2.78150905363230616e-01);
        assert_eq!(q[0][1], 1.50018830924020363e-02);
        assert_eq!(q[0][2], 2.80421646326272644e-02);
        // Row 26 is X, the fully degenerate amino code (Kp-3); it exercises
        // set_degenerate_probs over all 20 canonical residues.
        assert_eq!(q[26][0], 7.96249800736142377e-02);
        assert_eq!(q[26][19], 2.90649683917996587e-02);
    }

    #[test]
    fn pam30_matches_c_bitwise() {
        let (lambda, _q, _s) = conditionals(AlphabetType::Amino, "PAM30");
        assert_eq!(lambda, 3.37378055649468156e-01);
    }

    /// The nucleotide path the issue is about: DNA1 is the default built-in for
    /// non-amino alphabets (p7_builder.c:299).
    #[test]
    fn dna1_matches_c_bitwise() {
        let (lambda, q, s) = conditionals(AlphabetType::Dna, "DNA1");
        assert_eq!(s.k, 4);
        assert_eq!(s.kp, 18);
        assert_eq!(s.max(), 46);
        assert_eq!(s.nc, 15);
        assert_eq!(s.outorder, "ACGTRYMKSWHBVDN");
        assert_eq!(lambda, 1.99463435772605363e-02);
        assert_eq!(
            &q[0][..4],
            &[
                5.68593694999839916e-01,
                1.32566317464099276e-01,
                1.49419993768030446e-01,
                1.49419993768030446e-01
            ][..]
        );
        assert_eq!(
            &q[3][..4],
            &[
                1.47967491223543024e-01,
                1.77063723567418480e-01,
                1.33922446889969654e-01,
                5.41046338319068787e-01
            ][..]
        );
        // Row 15 is N (Kp-3), the fully degenerate nucleotide code.
        assert_eq!(
            &q[15][..4],
            &[
                2.49025627256058729e-01,
                2.42884466660844206e-01,
                2.56619752977197857e-01,
                2.51470153105899263e-01
            ][..]
        );
    }

    /// RNA uses the same DNA1 table and uniform background, so it must land on
    /// exactly the same numbers as DNA.
    #[test]
    fn rna_matches_dna_exactly() {
        let (ld, qd, _) = conditionals(AlphabetType::Dna, "DNA1");
        let (lr, qr, _) = conditionals(AlphabetType::Rna, "DNA1");
        assert_eq!(ld, lr);
        assert_eq!(qd, qr);
    }

    #[test]
    fn unknown_builtin_is_not_found() {
        let abc = Alphabet::new(AlphabetType::Amino);
        let mut s = ScoreMatrix::create(&abc);
        assert!(matches!(
            s.set("NOSUCHMATRIX", &abc),
            Err(ScoreMatrixError::NotFound(_))
        ));
        // DNA1 is a nucleotide built-in; it is not available for amino.
        assert!(matches!(
            s.set("DNA1", &abc),
            Err(ScoreMatrixError::NotFound(_))
        ));
        // ...and BLOSUM62 is not available for DNA.
        let dna = Alphabet::new(AlphabetType::Dna);
        let mut s = ScoreMatrix::create(&dna);
        assert!(matches!(
            s.set("BLOSUM62", &dna),
            Err(ScoreMatrixError::NotFound(_))
        ));
    }

    /// `esl_scorematrix_Read` round-trip: writing DNA1's canonical block out in
    /// matrix-file form and reading it back must reproduce those scores.
    #[test]
    fn read_parses_labelled_matrix_file() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = "# a comment\n\
                    \x20  A   C   G   T\n\
                    A  41 -32 -26 -26\n\
                    C -32  39 -38 -17\n\
                    G -26 -38  46 -31\n\
                    T -26 -17 -31  39\n";
        let s = ScoreMatrix::read(text, Some("/tmp/custom-dna.mx"), &abc).expect("read");
        assert_eq!(s.nc, 4);
        assert_eq!(s.outorder, "ACGT");
        assert_eq!(s.s[0][0], 41);
        assert_eq!(s.s[2][2], 46);
        assert_eq!(s.s[1][3], -17);
        assert_eq!(s.name.as_deref(), Some("custom-dna.mx"));
        assert_eq!(s.path.as_deref(), Some("/tmp/custom-dna.mx"));
    }

    #[test]
    fn read_rejects_missing_canonical_column() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let text = "  A   C   G\nA 1 0 0\nC 0 1 0\nG 0 0 1\n";
        let err = ScoreMatrix::read(text, None, &abc).unwrap_err();
        assert!(
            matches!(&err, ScoreMatrixError::Invalid(m) if m.contains("Expected to see a column")),
            "unexpected: {err}"
        );
    }
}
