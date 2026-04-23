//! P7_TRACE — Viterbi traceback path through the HMM.
//! Port of p7_trace.c and generic_vtrace.c.

use crate::alphabet::{Alphabet, Dsq};
use crate::dp::gmx::*;
use crate::profile::*;

/// State types in a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    S = 4,  // Start
    N = 5,  // N-terminal
    B = 6,  // Begin
    M = 1,  // Match
    D = 2,  // Delete
    I = 3,  // Insert
    E = 7,  // End
    C = 8,  // C-terminal
    T = 9,  // Terminal
    J = 10, // Join (multi-domain)
    X = 11, // Missing data / fragment marker
}

/// A traceback path through the HMM.
#[derive(Debug, Clone)]
pub struct Trace {
    /// State type at each position
    pub st: Vec<State>,
    /// Model node index (1..M for M/D/I, 0 otherwise)
    pub k: Vec<usize>,
    /// Sequence position (1..L for emitting states, 0 otherwise)
    pub i: Vec<usize>,
    /// Trace length
    pub n: usize,
    /// Model length for traces constructed relative to a model/MSA
    pub m: usize,
    /// Sequence/alignment coordinate length
    pub l: usize,
    /// Optional posterior probability annotation, parallel to state path.
    pub pp: Option<Vec<f32>>,
}

impl Trace {
    pub fn new() -> Self {
        Trace {
            st: Vec::new(),
            k: Vec::new(),
            i: Vec::new(),
            n: 0,
            m: 0,
            l: 0,
            pp: None,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.st.clear();
        self.k.clear();
        self.i.clear();
        if let Some(pp) = &mut self.pp {
            pp.clear();
        }
        self.n = 0;
    }

    #[inline(always)]
    pub fn append(&mut self, state: State, k: usize, i: usize) {
        self.append_internal(state, k, i, None);
    }

    #[inline(always)]
    pub fn append_with_pp(&mut self, state: State, k: usize, i: usize, pp: f32) {
        self.append_internal(state, k, i, Some(pp));
    }

    #[inline(always)]
    fn append_internal(&mut self, state: State, k: usize, i: usize, pp: Option<f32>) {
        self.st.push(state);
        match state {
            State::N | State::C | State::J => {
                let emitted = self.st.len() > 1 && self.st[self.st.len() - 2] == state;
                self.k.push(0);
                self.i.push(if emitted { i } else { 0 });
            }
            State::S | State::B | State::E | State::T => {
                self.k.push(0);
                self.i.push(0);
            }
            State::X => {
                self.k.push(k);
                self.i.push(i);
            }
            State::D => {
                self.k.push(k);
                self.i.push(0);
            }
            State::M | State::I => {
                self.k.push(k);
                self.i.push(i);
            }
        }
        if let Some(pp_value) = pp {
            if self.pp.is_none() {
                self.pp = Some(vec![0.0; self.n]);
            }
            self.pp.as_mut().unwrap().push(pp_value);
        } else if let Some(pp_values) = &mut self.pp {
            pp_values.push(0.0);
        }
        self.n += 1;
    }

    /// Find domain boundaries: returns (hmmfrom, hmmto, sqfrom, sqto) for first domain.
    pub fn domain_coords(&self) -> Option<(usize, usize, usize, usize)> {
        let mut hmmfrom = 0;
        let mut hmmto = 0;
        let mut sqfrom = 0;
        let mut sqto = 0;
        let mut in_domain = false;

        for z in 0..self.n {
            if self.st[z] == State::B {
                in_domain = true;
            }
            if in_domain && self.st[z] == State::M {
                if hmmfrom == 0 {
                    hmmfrom = self.k[z];
                    sqfrom = self.i[z];
                }
                hmmto = self.k[z];
                sqto = self.i[z];
            }
            if self.st[z] == State::E {
                break;
            }
        }

        if hmmfrom > 0 {
            Some((hmmfrom, hmmto, sqfrom, sqto))
        } else {
            None
        }
    }
}

/// Viterbi traceback: reconstruct the optimal path through a filled DP matrix.
/// Returns a Trace with the state path.
pub fn g_trace(dsq: &[Dsq], l: usize, gm: &Profile, gx: &Gmx) -> Trace {
    let m = gm.m;
    let mut tr = Trace::new();
    let tol = 1e-5_f32;

    // Start from terminal state
    tr.append(State::T, 0, 0);

    // C state at position L
    tr.append(State::C, 0, l);
    let mut i = l;
    let mut cur_state = State::C;

    // Walk backwards through the trace
    loop {
        match cur_state {
            State::C => {
                // C can come from C(i-1) or E(i)
                if i > 0 {
                    let c_from_e = gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE];
                    if (gx.xmx(i, P7G_C) - c_from_e).abs() < tol {
                        cur_state = State::E;
                        tr.append(State::E, 0, i);
                    } else {
                        i -= 1;
                        tr.append(State::C, 0, i);
                    }
                } else {
                    // At i=0, must be S->N->...->C
                    cur_state = State::E;
                    tr.append(State::E, 0, 0);
                }
            }
            State::E => {
                // E comes from M_k or D_M — find which k
                let mut found_k = 0;
                for k in (1..=m).rev() {
                    if (gx.xmx(i, P7G_E) - gx.mmx(i, k)).abs() < tol {
                        found_k = k;
                        break;
                    }
                    if k == m && (gx.xmx(i, P7G_E) - gx.dmx(i, m)).abs() < tol {
                        found_k = m;
                        cur_state = State::D;
                        tr.append(State::D, m, 0);
                        break;
                    }
                }
                if cur_state != State::D {
                    if found_k > 0 {
                        cur_state = State::M;
                        tr.append(State::M, found_k, i);
                    } else {
                        // Fallback: find best-scoring M_k
                        let mut best_k = m;
                        let mut best_sc = f32::NEG_INFINITY;
                        for k in 1..=m {
                            if gx.mmx(i, k) > best_sc {
                                best_sc = gx.mmx(i, k);
                                best_k = k;
                            }
                        }
                        cur_state = State::M;
                        tr.append(State::M, best_k, i);
                    }
                }
            }
            State::M => {
                let k = *tr.k.last().unwrap();
                if k == 0 || i == 0 {
                    // Transition to B
                    cur_state = State::B;
                    tr.append(State::B, 0, i);
                    continue;
                }
                // M(i,k) can come from B, M(i-1,k-1), I(i-1,k-1), D(i-1,k-1)
                if k > 1 {
                    let mm_sc = gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM);
                    let im_sc = gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM);
                    let dm_sc = gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM);

                    let sc = gx.mmx(i, k) - gm.msc(k, dsq[i] as usize);
                    if (sc - mm_sc).abs() < tol {
                        i -= 1;
                        cur_state = State::M;
                        tr.append(State::M, k - 1, i);
                    } else if (sc - im_sc).abs() < tol {
                        i -= 1;
                        cur_state = State::I;
                        tr.append(State::I, k - 1, i);
                    } else if (sc - dm_sc).abs() < tol {
                        i -= 1;
                        cur_state = State::D;
                        tr.append(State::D, k - 1, 0);
                    } else {
                        // B->M entry
                        i -= 1;
                        cur_state = State::B;
                        tr.append(State::B, 0, i);
                    }
                } else {
                    // k=1: must come from B
                    i -= 1;
                    cur_state = State::B;
                    tr.append(State::B, 0, i);
                }
            }
            State::D => {
                let k = *tr.k.last().unwrap();
                if k <= 1 {
                    cur_state = State::B;
                    tr.append(State::B, 0, i);
                    continue;
                }
                // D(i,k) from M(i,k-1) or D(i,k-1)
                let md_sc = gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD);
                if (gx.dmx(i, k) - md_sc).abs() < tol {
                    cur_state = State::M;
                    tr.append(State::M, k - 1, i);
                } else {
                    cur_state = State::D;
                    tr.append(State::D, k - 1, 0);
                }
            }
            State::I => {
                let k = *tr.k.last().unwrap();
                i -= 1;
                // I(i,k) from M(i-1,k) or I(i-1,k)
                let mi_sc = gx.mmx(i, k) + gm.tsc(k, P7P_MI);
                let sc = gx.imx(i + 1, k) - gm.isc(k, dsq[i + 1] as usize);
                if (sc - mi_sc).abs() < tol {
                    cur_state = State::M;
                    tr.append(State::M, k, i);
                } else {
                    cur_state = State::I;
                    tr.append(State::I, k, i);
                }
            }
            State::B => {
                // B from N or J
                let bn_sc = gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE];
                if (gx.xmx(i, P7G_B) - bn_sc).abs() < tol {
                    cur_state = State::N;
                    tr.append(State::N, 0, i);
                } else {
                    cur_state = State::J;
                    tr.append(State::J, 0, i);
                }
            }
            State::N => {
                if i == 0 {
                    tr.append(State::S, 0, 0);
                    break;
                }
                // N from N(i-1) or S
                let nn_sc = gx.xmx(i - 1, P7G_N) + gm.xsc[P7P_N][P7P_LOOP];
                if (gx.xmx(i, P7G_N) - nn_sc).abs() < tol && i > 0 {
                    i -= 1;
                    tr.append(State::N, 0, i);
                } else {
                    tr.append(State::S, 0, 0);
                    break;
                }
            }
            State::J => {
                // J from J(i-1) or E(i)
                if i > 0 {
                    let je_sc = gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP];
                    if (gx.xmx(i, P7G_J) - je_sc).abs() < tol {
                        cur_state = State::E;
                        tr.append(State::E, 0, i);
                    } else {
                        i -= 1;
                        tr.append(State::J, 0, i);
                    }
                } else {
                    cur_state = State::E;
                    tr.append(State::E, 0, 0);
                }
            }
            _ => break,
        }

        // Safety: prevent infinite loops
        if tr.n > l + m + 100 {
            break;
        }
    }

    // Reverse the trace (we built it backwards)
    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();

    tr
}

/// Generate alignment display strings from a trace.
/// If `pp` (posterior probability matrix) is provided, use real PP values.
pub fn alignment_display(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
) -> Option<AlignmentDisplay> {
    alignment_display_with_pp(tr, dsq, hmm, abc, None)
}

/// Generate alignment display with optional posterior probabilities.
pub fn alignment_display_with_pp(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
    pp: Option<&crate::dp::gmx::Gmx>,
) -> Option<AlignmentDisplay> {
    alignment_display_with_pp_bg(tr, dsq, hmm, abc, pp, None)
}

/// Like `alignment_display_with_pp` but with an override for the per-residue
/// background frequencies. Used in long_target mode where the model is
/// reparameterized: C's p7_alidisplay_Create compares `FGetEmission > 1.0`
/// (i.e. emission / reparameterized_bg > 1), so Rust's mline `+` threshold
/// needs the same reparameterized bg to be byte-identical.
pub fn alignment_display_with_pp_bg(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
    pp: Option<&crate::dp::gmx::Gmx>,
    bg_override: Option<&[f32]>,
) -> Option<AlignmentDisplay> {
    // Find first and last M states (domain boundaries)
    let mut z1 = None;
    let mut z2 = None;
    for z in 0..tr.n {
        if tr.st[z] == State::M || tr.st[z] == State::D || tr.st[z] == State::I {
            if z1.is_none() {
                z1 = Some(z);
            }
            z2 = Some(z);
        }
    }

    let z1 = z1?;
    let z2 = z2?;

    let mut model = String::new();
    let mut mline = String::new();
    let mut aseq = String::new();
    let mut ppline = String::new();
    let mut rfline = String::new();
    let has_rf = hmm
        .rf
        .as_ref()
        .map_or(false, |rf| !rf.is_empty() && rf[0] != 0);
    let mut hmmfrom = 0;
    let mut hmmto = 0;
    let mut sqfrom = 0;
    let mut sqto = 0;

    for z in z1..=z2 {
        match tr.st[z] {
            State::M => {
                let k = tr.k[z];
                let i = tr.i[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                    sqfrom = i;
                }
                hmmto = k;
                sqto = i;
                if has_rf {
                    let rf = hmm.rf.as_ref().unwrap();
                    rfline.push(if k < rf.len() { rf[k] as char } else { '.' });
                }

                // Model consensus
                let cons_ch = if let Some(ref cons) = hmm.consensus {
                    if k < cons.len() {
                        cons[k] as char
                    } else {
                        'x'
                    }
                } else {
                    'x'
                };
                model.push(cons_ch);

                // Target sequence
                let seq_ch = if i > 0 && (dsq[i] as usize) < abc.kp {
                    abc.sym[dsq[i] as usize] as char
                } else {
                    '?'
                };
                aseq.push(seq_ch);

                // Match line: C HMMER (p7_alidisplay.c:220-224) pushes the
                // model character on identity, `+` when emission-odds ratio
                // exceeds 1.0 (positive log-odds), space otherwise.
                if cons_ch.to_ascii_uppercase() == seq_ch.to_ascii_uppercase() {
                    mline.push(cons_ch);
                } else {
                    let x = dsq[i] as usize;
                    let sc = hmm
                        .mat
                        .get(k)
                        .and_then(|row| row.get(x))
                        .copied()
                        .unwrap_or(0.0);
                    let bg = if let Some(bg_f) = bg_override {
                        bg_f.get(x).copied().unwrap_or(0.25)
                    } else if hmm.abc_k == 4 {
                        0.25_f32
                    } else {
                        crate::bg::AMINO_FREQUENCIES.get(x).copied().unwrap_or(0.05)
                    };
                    if x < hmm.abc_k && sc > bg {
                        mline.push('+');
                    } else {
                        mline.push(' ');
                    }
                }
                // PP for match state
                if let Some(pp_mx) = pp {
                    let pp_val = pp_mx.mmx(i, k);
                    ppline.push(crate::dp::generic_optacc::pp_to_char(pp_val.min(1.0)));
                } else {
                    ppline.push('*');
                }
            }
            State::I => {
                let i = tr.i[z];
                let k = tr.k[z];
                if has_rf {
                    rfline.push('.');
                }
                model.push('.');
                let seq_ch = if i > 0 && (dsq[i] as usize) < abc.kp {
                    (abc.sym[dsq[i] as usize] as char).to_ascii_lowercase()
                } else {
                    'x'
                };
                aseq.push(seq_ch);
                mline.push(' ');
                // PP for insert state: C HMMER uses the I-state posterior at
                // (i, k). We look at pp.imx(i, k) when available, else '.'.
                if let Some(pp_mx) = pp {
                    let pp_val = pp_mx.imx(i, k);
                    ppline.push(crate::dp::generic_optacc::pp_to_char(pp_val.min(1.0)));
                } else {
                    ppline.push('.');
                }
                if sqto == 0 && sqfrom == 0 {
                    sqfrom = i;
                }
                sqto = i;
            }
            State::D => {
                let k = tr.k[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                }
                hmmto = k;
                if has_rf {
                    let rf = hmm.rf.as_ref().unwrap();
                    rfline.push(if k < rf.len() { rf[k] as char } else { '.' });
                }

                let cons_ch = if let Some(ref cons) = hmm.consensus {
                    if k < cons.len() {
                        cons[k] as char
                    } else {
                        'x'
                    }
                } else {
                    'x'
                };
                model.push(cons_ch);
                aseq.push('-');
                mline.push(' ');
                ppline.push('.');
            }
            _ => {}
        }
    }

    Some(AlignmentDisplay {
        model,
        mline,
        aseq,
        ppline,
        rfline,
        hmmfrom,
        hmmto,
        sqfrom,
        sqto,
    })
}

/// Compute the same coordinate span as `alignment_display_with_pp` without
/// constructing printable alignment strings.
pub fn alignment_coords(tr: &Trace) -> Option<(usize, usize, usize, usize)> {
    let mut z1 = None;
    let mut z2 = None;
    for z in 0..tr.n {
        if tr.st[z] == State::M || tr.st[z] == State::D || tr.st[z] == State::I {
            if z1.is_none() {
                z1 = Some(z);
            }
            z2 = Some(z);
        }
    }

    let z1 = z1?;
    let z2 = z2?;
    let mut hmmfrom = 0;
    let mut hmmto = 0;
    let mut sqfrom = 0;
    let mut sqto = 0;

    for z in z1..=z2 {
        match tr.st[z] {
            State::M => {
                let k = tr.k[z];
                let i = tr.i[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                    sqfrom = i;
                }
                hmmto = k;
                sqto = i;
            }
            State::I => {
                let i = tr.i[z];
                if sqto == 0 && sqfrom == 0 {
                    sqfrom = i;
                }
                sqto = i;
            }
            State::D => {
                let k = tr.k[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                }
                hmmto = k;
            }
            _ => {}
        }
    }

    Some((hmmfrom, hmmto, sqfrom, sqto))
}

/// Alignment display data for one domain.
pub struct AlignmentDisplay {
    pub model: String,
    pub mline: String,
    pub ppline: String,
    pub aseq: String,
    pub rfline: String,
    pub hmmfrom: usize,
    pub hmmto: usize,
    pub sqfrom: usize,
    pub sqto: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::dp::generic_viterbi::g_viterbi;
    use crate::dp::gmx::Gmx;
    use std::path::Path;

    #[test]
    fn test_traceback_basic() {
        crate::logsum::p7_flogsuminit();
        let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        let mut gx = Gmx::new(gm.m, l);
        g_viterbi(&dsq, l, &gm, &mut gx);

        let tr = g_trace(&dsq, l, &gm, &gx);
        assert!(tr.n > 0, "Trace should be non-empty");

        // Should contain S...B...M...E...C...T
        assert_eq!(tr.st[0], State::S);
        assert_eq!(*tr.st.last().unwrap(), State::T);

        // Check domain coords
        let coords = tr.domain_coords();
        assert!(coords.is_some());
        let (hf, ht, sf, st) = coords.unwrap();
        assert!(hf >= 1 && ht <= 20);
        assert!(sf >= 1 && st <= 20);
    }

    #[test]
    fn test_alignment_display() {
        crate::logsum::p7_flogsuminit();
        let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa.hmm"
        )))
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;

        let mut gx = Gmx::new(gm.m, l);
        g_viterbi(&dsq, l, &gm, &mut gx);

        let tr = g_trace(&dsq, l, &gm, &gx);
        let ad = alignment_display(&tr, &dsq, &hmm, &abc);
        assert!(ad.is_some());
        let ad = ad.unwrap();
        assert!(!ad.model.is_empty());
        assert!(!ad.aseq.is_empty());
        assert_eq!(ad.model.len(), ad.aseq.len());
    }
}
