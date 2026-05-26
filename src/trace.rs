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
    /// Allocate a new empty (growable, reusable) traceback.
    /// Port of `p7_trace_Create()`; PP storage is allocated lazily on the first
    /// `append_with_pp()` call (matching `p7_trace_CreateWithPP()`).
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

    /// Reinitialize the trace, retaining allocations for reuse.
    /// Port of `p7_trace_Reuse()`.
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

    /// Append one state (no PP) to a left-to-right-growing trace.
    /// Port of `p7_trace_Append()`. For emit-on-transition states (N/C/J), the
    /// first state in a run is recorded with i=0; subsequent ones carry `i`.
    #[inline(always)]
    pub fn append(&mut self, state: State, k: usize, i: usize) {
        self.append_internal(state, k, i, None);
    }

    /// Append one state with an associated posterior probability for emitted
    /// residues. Port of `p7_trace_AppendWithPP()`; lazily allocates `self.pp`.
    /// `pp` is silently ignored for nonemitting states (recorded as 0.0).
    #[inline(always)]
    pub fn append_with_pp(&mut self, state: State, k: usize, i: usize, pp: f32) {
        self.append_internal(state, k, i, Some(pp));
    }

    /// Shared implementation backing both `append` and `append_with_pp`.
    /// Encodes the state-specific rules for which of (k, i, pp) get stored.
    #[inline(always)]
    fn append_internal(&mut self, state: State, k: usize, i: usize, pp: Option<f32>) {
        self.st.push(state);
        // Per-state stored pp (C zeroes pp for all nonemitting states, and for
        // the first N/C/J in a run; see p7_trace_AppendWithPP, p7_trace.c:1056-1085).
        let mut stored_pp = pp.unwrap_or(0.0);
        match state {
            State::N | State::C | State::J => {
                let emitted = self.st.len() > 1 && self.st[self.st.len() - 2] == state;
                self.k.push(0);
                if emitted {
                    self.i.push(i);
                } else {
                    // First N/C/J in a run is non-emitting: i=0, pp=0.0.
                    self.i.push(0);
                    stored_pp = 0.0;
                }
            }
            State::S | State::B | State::E | State::T => {
                self.k.push(0);
                self.i.push(0);
                stored_pp = 0.0;
            }
            State::X => {
                self.k.push(k);
                self.i.push(i);
                stored_pp = 0.0;
            }
            State::D => {
                self.k.push(k);
                self.i.push(0);
                stored_pp = 0.0;
            }
            State::M | State::I => {
                self.k.push(k);
                self.i.push(i);
            }
        }
        if pp.is_some() {
            if self.pp.is_none() {
                self.pp = Some(vec![0.0; self.n]);
            }
            self.pp.as_mut().unwrap().push(stored_pp);
        } else if let Some(pp_values) = &mut self.pp {
            pp_values.push(0.0);
        }
        self.n += 1;
    }

    /// Return the (hmmfrom, hmmto, sqfrom, sqto) bounds of the first domain
    /// (states between the first B and the next E). Port of
    /// `p7_trace_GetDomainCoords()` restricted to the first domain.
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

/// Float near-equality test. Faithful port of `esl_FCompare_old`
/// (hmmer/easel/easel.c:2400). Used by the Viterbi traceback to compare a DP
/// cell value against `prev + tsc + emission` candidate predecessors.
#[inline]
fn esl_fcompare_old(a: f32, b: f32, tol: f32) -> bool {
    if a.is_infinite() && b.is_infinite() {
        return true;
    }
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    if a == b {
        return true;
    }
    if a.abs() == 0.0 && b.abs() <= tol {
        return true;
    }
    if b.abs() == 0.0 && a.abs() <= tol {
        return true;
    }
    if 2.0 * (a - b).abs() / (a + b).abs() <= tol {
        return true;
    }
    false
}

/// Viterbi traceback: walk a filled generic Viterbi DP matrix `gx` backwards
/// from T to S to recover the optimal state path. Faithful port of
/// `p7_GTrace()` in `hmmer/src/generic_vtrace.c`. Reconstructs an M+I+D path
/// plus the surrounding S/N/B/E/C/J/T scaffold and returns it forward-ordered.
///
/// This is a reconstruction traceback: predecessor scores are compared against
/// the full `prev + tsc + emission` sum in exactly the order Viterbi computed
/// them (B,M,I,D for M-states), per the J1/121 note in the C source, so the
/// near-equality tie-break matches C cell-for-cell.
pub fn g_trace(dsq: &[Dsq], l: usize, gm: &Profile, gx: &Gmx) -> Trace {
    let m = gm.m;
    let mut tr = Trace::new();
    let tol = 1e-5_f32;

    let mut i = l; // position in seq (1..L)
    let mut k = 0usize; // position in model (1..M)

    tr.append(State::T, k, i);
    tr.append(State::C, k, i);
    let mut sprv = State::C;

    while sprv != State::S {
        // rsc[dsq[i]] available only for i>0; emission scores indexed with x=dsq[i].
        let scur: State = match sprv {
            State::C => {
                // C(i) comes from C(i-1) or E(i)
                if esl_fcompare_old(
                    gx.xmx(i, P7G_C),
                    gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
                    tol,
                ) {
                    State::C
                } else if esl_fcompare_old(
                    gx.xmx(i, P7G_C),
                    gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE],
                    tol,
                ) {
                    State::E
                } else {
                    // C couldn't be traced (impossible in a valid matrix).
                    break;
                }
            }
            State::E => {
                // E connects from any M state; k set here.
                if gm.is_local() {
                    // Can't come from D in a local Viterbi trace.
                    let mut found = 0usize;
                    let mut kk = m;
                    while kk >= 1 {
                        if esl_fcompare_old(gx.xmx(i, P7G_E), gx.mmx(i, kk), tol) {
                            found = kk;
                            break;
                        }
                        kk -= 1;
                    }
                    if found == 0 {
                        break; // E couldn't be traced
                    }
                    k = found;
                    State::M
                } else {
                    // glocal mode: we come from either M_M or D_M
                    if esl_fcompare_old(gx.xmx(i, P7G_E), gx.mmx(i, m), tol) {
                        k = m;
                        State::M
                    } else if esl_fcompare_old(gx.xmx(i, P7G_E), gx.dmx(i, m), tol) {
                        k = m;
                        State::D
                    } else {
                        break; // E couldn't be traced
                    }
                }
            }
            State::M => {
                // M connects from i-1,k-1, or B. Test B-entry FIRST, then MM, IM, DM,
                // comparing against the full prev+tsc+emission sum (J1/121).
                let msc = gm.msc(k, dsq[i] as usize);
                let s = if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM) + msc,
                    tol,
                ) {
                    State::B
                } else if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM) + msc,
                    tol,
                ) {
                    State::M
                } else if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM) + msc,
                    tol,
                ) {
                    State::I
                } else if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM) + msc,
                    tol,
                ) {
                    State::D
                } else {
                    break; // M couldn't be traced
                };
                k -= 1;
                i -= 1;
                s
            }
            State::D => {
                // D connects from M,D at i,k-1
                let s = if esl_fcompare_old(
                    gx.dmx(i, k),
                    gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD),
                    tol,
                ) {
                    State::M
                } else if esl_fcompare_old(
                    gx.dmx(i, k),
                    gx.dmx(i, k - 1) + gm.tsc(k - 1, P7P_DD),
                    tol,
                ) {
                    State::D
                } else {
                    break; // D couldn't be traced
                };
                k -= 1;
                s
            }
            State::I => {
                // I connects from M,I at i-1,k
                let isc = gm.isc(k, dsq[i] as usize);
                let s = if esl_fcompare_old(
                    gx.imx(i, k),
                    gx.mmx(i - 1, k) + gm.tsc(k, P7P_MI) + isc,
                    tol,
                ) {
                    State::M
                } else if esl_fcompare_old(
                    gx.imx(i, k),
                    gx.imx(i - 1, k) + gm.tsc(k, P7P_II) + isc,
                    tol,
                ) {
                    State::I
                } else {
                    break; // I couldn't be traced
                };
                i -= 1;
                s
            }
            State::N => {
                // N connects from S, N
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::B => {
                // B connects from N, J
                if esl_fcompare_old(
                    gx.xmx(i, P7G_B),
                    gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE],
                    tol,
                ) {
                    State::N
                } else if esl_fcompare_old(
                    gx.xmx(i, P7G_B),
                    gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE],
                    tol,
                ) {
                    State::J
                } else {
                    break; // B couldn't be traced
                }
            }
            State::J => {
                // J connects from E(i) or J(i-1)
                if esl_fcompare_old(
                    gx.xmx(i, P7G_J),
                    gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
                    tol,
                ) {
                    State::J
                } else if esl_fcompare_old(
                    gx.xmx(i, P7G_J),
                    gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP],
                    tol,
                ) {
                    State::E
                } else {
                    break; // J couldn't be traced
                }
            }
            _ => break, // bogus state
        };

        // Append this state and the current i,k to be explained.
        tr.append(scur, k, i);

        // For NCJ, we had to defer the i decrement.
        if (scur == State::N || scur == State::J || scur == State::C) && scur == sprv {
            i -= 1;
        }

        sprv = scur;
    }

    tr.m = gm.m;
    tr.l = l;

    // Faithful port of p7_trace_Reverse (p7_trace.c:1108-1151).
    // For emit-on-transition N/C/J runs built backwards, pull residues back by
    // one (C-,Cx,Cx,Cx -> Cx,Cx,Cx,C-) before the in-place reversal.
    if tr.n > 0 {
        for z in 0..(tr.n - 1) {
            let run = (tr.st[z] == State::N && tr.st[z + 1] == State::N)
                || (tr.st[z] == State::C && tr.st[z + 1] == State::C)
                || (tr.st[z] == State::J && tr.st[z + 1] == State::J);
            if run && tr.i[z] == 0 && tr.i[z + 1] > 0 {
                tr.i[z] = tr.i[z + 1];
                tr.i[z + 1] = 0;
                if let Some(pp) = &mut tr.pp {
                    pp[z] = pp[z + 1];
                    pp[z + 1] = 0.0;
                }
            }
        }
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    if let Some(pp) = &mut tr.pp {
        pp.reverse();
    }

    tr
}

/// Build the printable alignment-display strings (model, mline, aseq, ppline,
/// rfline) for the first domain in `tr`. Equivalent to a minimal
/// `p7_alidisplay_Create()` call. Use `alignment_display_with_pp` for real PPs.
pub fn alignment_display(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
) -> Option<AlignmentDisplay> {
    alignment_display_with_pp(tr, dsq, hmm, abc, None)
}

/// Like `alignment_display`, but uses the supplied posterior-probability
/// matrix to emit a real ppline. Port of `p7_alidisplay_Create()` with PPs.
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
    alignment_display_with_pp_emission_odds(tr, dsq, hmm, abc, pp, bg_override, None)
}

/// Build an alignment display, optionally using optimized-profile emission
/// odds for C-style match-line `+` decisions.
pub fn alignment_display_with_pp_emission_odds(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
    pp: Option<&crate::dp::gmx::Gmx>,
    bg_override: Option<&[f32]>,
    emission_odds: Option<&dyn Fn(usize, usize) -> Option<f32>>,
) -> Option<AlignmentDisplay> {
    // C anchors the display window on M states ONLY (p7_alidisplay.c:108-119):
    // z1 = first p7T_M state, z2 = last p7T_M state. Interior D/I between them
    // are still emitted, but leading/trailing D and I (e.g. a `...M D D E`
    // delete-trailer) are excluded so they don't leak into the display or
    // over-advance hmmto/sqto.
    let mut z1 = None;
    let mut z2 = None;
    for z in 0..tr.n {
        if tr.st[z] == State::M {
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
                    let positive_emission =
                        if let Some(get_odds) = emission_odds.filter(|_| hmm.abc_k > 4) {
                            get_odds(k, x)
                                .map(|odds| odds > 1.0)
                                .unwrap_or(x < hmm.abc_k && sc > bg)
                        } else {
                            x < hmm.abc_k && sc > bg
                        };
                    if positive_emission {
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
    // M-only anchoring, matching C's p7_alidisplay_Create (z1 = first M,
    // z2 = last M). Leading/trailing D and I are excluded.
    let mut z1 = None;
    let mut z2 = None;
    for z in 0..tr.n {
        if tr.st[z] == State::M {
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

    /// Sanity check: a Viterbi traceback over the 20aa toy HMM produces a
    /// well-formed S...T trace covering plausible model and sequence coords.
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

    /// `alignment_display` produces nonempty model/aseq strings of equal length
    /// for a basic 20aa Viterbi traceback.
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
