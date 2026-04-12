//! P7_OPROFILE - SSE2-optimized scoring profile.
//! Direct port of impl_sse/p7_oprofile.c.

use crate::profile::*;

/// Number of __m128i vectors needed for byte-precision (MSV): ceil(M/16), min 2
pub fn nqb(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 16) + 1)
}

/// Number of __m128i vectors needed for word-precision (Viterbi): ceil(M/8), min 2
pub fn nqw(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 8) + 1)
}

/// Number of __m128 vectors needed for float-precision (Forward): ceil(M/4), min 2
pub fn nqf(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 4) + 1)
}

// Viterbi transition order in twv
pub const P7O_BM: usize = 0;
pub const P7O_MM: usize = 1;
pub const P7O_IM: usize = 2;
pub const P7O_DM: usize = 3;
pub const P7O_MD: usize = 4;
pub const P7O_MI: usize = 5;
pub const P7O_II: usize = 6;
pub const P7O_DD: usize = 7;

// Special states
pub const P7O_E: usize = 0;
pub const P7O_N: usize = 1;
pub const P7O_J: usize = 2;
pub const P7O_C: usize = 3;
pub const P7O_NXSTATES: usize = 4;
pub const P7O_MOVE: usize = 0;
pub const P7O_LOOP: usize = 1;
pub const P7O_NXTRANS: usize = 2;

/// Optimized profile for SSE2 SIMD operations.
#[derive(Debug, Clone)]
pub struct OProfile {
    // === MSV byte-precision fields ===
    /// MSV byte-precision match scores: rbv[residue_code][q_vector][16 bytes]
    pub rbv: Vec<Vec<[u8; 16]>>,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,

    // === Viterbi word-precision fields ===
    /// Word-precision match emission scores: rwv[residue_code][q_vector][8 words]
    pub rwv: Vec<Vec<[i16; 8]>>,
    /// Word-precision transition scores: twv[j][8 words]
    /// Layout: for each q, 7 transitions (BM,MM,IM,DM,MD,MI,II), then DD at end
    pub twv: Vec<[i16; 8]>,
    /// Special state word scores: xw[state][transition]
    pub xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub ncj_roundoff: f32,

    // === Forward/Backward float-precision fields ===
    /// Float-precision emission scores (probability ratios): rfv[residue][q_vector][4 floats]
    pub rfv: Vec<Vec<[f32; 4]>>,
    /// Float-precision transition scores: tfv[j][4 floats]
    pub tfv: Vec<[f32; 4]>,
    /// Special state float scores: xf[state][transition]
    pub xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],

    // === Common fields ===
    pub m: usize,
    pub l: i32,
    pub nj: f32,
    pub mode: i32,
    pub evparam: [f32; 6],
    pub cutoff: [f32; 6],
    pub compo: [f32; 20],
    pub name: String,
    pub abc_k: usize,
    pub abc_kp: usize,
}

/// Convert a float log-odds score to biased byte representation.
/// Matches C behavior: result = (uint8_t)(-round(scale * sc)) + bias
/// For positive scores, -round(scale * sc) is negative, and the unsigned wrapping
/// produces the correct value.
fn biased_byteify(scale_b: f32, bias_b: u8, sc: f32) -> u8 {
    let negated = -1.0 * (scale_b * sc).round();
    if negated > (255 - bias_b) as f32 {
        255
    } else {
        // Match C's unsigned wrap behavior: cast to i32 first, then wrapping add
        let ival = negated as i32;
        (ival as u8).wrapping_add(bias_b)
    }
}

/// Convert a float score to word (int16) representation.
fn wordify(scale_w: f32, sc: f32) -> i16 {
    let sc = (scale_w * sc).round();
    if sc >= 32767.0 {
        32767
    } else if sc <= -32768.0 {
        -32768
    } else {
        sc as i16
    }
}

/// Convert a float log-odds score to unbiased byte representation.
fn unbiased_byteify(scale_b: f32, sc: f32) -> u8 {
    let negated = -1.0 * (scale_b * sc).round();
    if negated > 255.0 {
        255
    } else {
        negated as i32 as u8
    }
}

impl OProfile {
    /// Convert a generic Profile to an optimized OProfile for SSE2.
    pub fn convert(gm: &Profile) -> Self {
        let m = gm.m;
        let nq = nqb(m);
        let kp = gm.abc_kp;
        let k = gm.abc_k;

        // Determine scale and bias for MSV byte scores
        let scale_b = 3.0_f32 / std::f32::consts::LN_2;
        let base_b: u8 = 190;

        // Find maximum match emission score across all residues and positions
        let mut max_sc = 0.0_f32;
        for x in 0..k {
            for node in 1..=m {
                let sc = gm.msc(node, x);
                if sc.is_finite() && sc > max_sc {
                    max_sc = sc;
                }
            }
        }

        let bias_b = unbiased_byteify(scale_b, -1.0 * max_sc);

        // Build striped MSV match score vectors
        let mut rbv = vec![vec![[0u8; 16]; nq]; kp];

        for x in 0..kp {
            for q in 0..nq {
                let mut tmp = [0u8; 16];
                for z in 0..16 {
                    let node = q + 1 + z * nq;
                    if node <= m {
                        tmp[z] = biased_byteify(scale_b, bias_b, gm.msc(node, x));
                    } else {
                        tmp[z] = 255; // -infinity in offset arithmetic
                    }
                }
                rbv[x][q] = tmp;
            }
        }

        // Transition costs
        let tbm_b = unbiased_byteify(
            scale_b,
            (2.0_f32 / (m as f32 * (m as f32 + 1.0))).ln(),
        );
        let tec_b = unbiased_byteify(scale_b, 0.5_f32.ln());
        let tjb_b = unbiased_byteify(
            scale_b,
            (3.0_f32 / (gm.l as f32 + 3.0)).ln(),
        );

        // === Viterbi word-precision conversion ===
        let scale_w = 500.0_f32 / std::f32::consts::LN_2;
        let base_w: i16 = 12000;
        let nqw = nqw(m);

        // Striped word match scores
        let mut rwv = vec![vec![[0i16; 8]; nqw]; kp];
        for x in 0..kp {
            let mut qi = 0;
            let mut ki = 1;
            while qi < nqw {
                let mut tmp = [0i16; 8];
                for z in 0..8 {
                    let node = ki + z * nqw;
                    tmp[z] = if node <= m {
                        wordify(scale_w, gm.msc(node, x))
                    } else {
                        -32768
                    };
                }
                rwv[x][qi] = tmp;
                qi += 1;
                ki += 1;
            }
        }

        // Transition scores (7 per q, then DD at end)
        let mut twv = vec![[0i16; 8]; 8 * nqw];
        let mut j = 0;
        for qi in 0..nqw {
            let ki = qi + 1;
            // 7 transitions: BM, MM, IM, DM, MD, MI, II
            let trans_specs: [(usize, usize, i16); 7] = [
                (P7P_BM, ki.wrapping_sub(1), 0),  // BM: off-by-one, starts from k=0
                (P7P_MM, ki.wrapping_sub(1), 0),  // MM: rotated by -1
                (P7P_IM, ki.wrapping_sub(1), 0),  // IM: rotated by -1
                (P7P_DM, ki.wrapping_sub(1), 0),  // DM: rotated by -1
                (P7P_MD, ki, 0),                    // MD: straight
                (P7P_MI, ki, 0),                    // MI: straight
                (P7P_II, ki, -1),                   // II: maxval=-1 (prevent zero-cost II)
            ];

            for &(tg, kb, maxval) in &trans_specs {
                let mut tmp = [0i16; 8];
                for z in 0..8 {
                    let node = kb + z * nqw;
                    let val = if node < m {
                        wordify(scale_w, gm.tsc(node, tg))
                    } else {
                        -32768
                    };
                    tmp[z] = if val >= maxval { maxval } else { val };
                }
                twv[j] = tmp;
                j += 1;
            }
        }
        // DD transitions at the end
        for qi in 0..nqw {
            let ki = qi + 1;
            let mut tmp = [0i16; 8];
            for z in 0..8 {
                let node = ki + z * nqw;
                tmp[z] = if node < m {
                    wordify(scale_w, gm.tsc(node, P7P_DD))
                } else {
                    -32768
                };
            }
            twv[j] = tmp;
            j += 1;
        }

        // Special state word scores
        let mut xw = [[0i16; P7O_NXTRANS]; P7O_NXSTATES];
        xw[P7O_E][P7O_LOOP] = wordify(scale_w, gm.xsc[P7P_E][P7P_LOOP]);
        xw[P7O_E][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_E][P7P_MOVE]);
        xw[P7O_N][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_N][P7P_MOVE]);
        xw[P7O_N][P7O_LOOP] = 0; // hardcoded: -3nat approximation
        xw[P7O_C][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_C][P7P_MOVE]);
        xw[P7O_C][P7O_LOOP] = 0;
        xw[P7O_J][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_J][P7P_MOVE]);
        xw[P7O_J][P7O_LOOP] = 0;

        // DD bound for lazy F evaluation
        let mut ddbound_w: i16 = -32768;
        for node in 2..m.saturating_sub(1) {
            let dd = wordify(scale_w, gm.tsc(node, P7P_DD)) as i32;
            let dm = wordify(scale_w, gm.tsc(node + 1, P7P_DM)) as i32;
            let bm = wordify(scale_w, gm.tsc(node + 1, P7P_BM)) as i32;
            let ddtmp = dd + dm - bm;
            if ddtmp > ddbound_w as i32 {
                ddbound_w = ddtmp.min(32767) as i16;
            }
        }

        // === Forward/Backward float-precision conversion ===
        let nqf = nqf(m);

        // Float emission scores (probability ratios = exp(log-odds scores))
        let mut rfv = vec![vec![[0.0f32; 4]; nqf]; kp];
        for x in 0..kp {
            let mut qi = 0;
            let mut ki = 1;
            while qi < nqf {
                let mut tmp = [0.0f32; 4];
                for z in 0..4 {
                    let node = ki + z * nqf;
                    tmp[z] = if node <= m {
                        gm.msc(node, x).exp()
                    } else {
                        0.0 // exp(-inf) = 0
                    };
                }
                rfv[x][qi] = tmp;
                qi += 1;
                ki += 1;
            }
        }

        // Float transition scores (probabilities = exp(log scores))
        let mut tfv = vec![[0.0f32; 4]; 8 * nqf];
        let mut j = 0;
        for qi in 0..nqf {
            let ki = qi + 1;
            let trans_specs: [(usize, usize); 7] = [
                (P7P_BM, ki.wrapping_sub(1)),
                (P7P_MM, ki.wrapping_sub(1)),
                (P7P_IM, ki.wrapping_sub(1)),
                (P7P_DM, ki.wrapping_sub(1)),
                (P7P_MD, ki),
                (P7P_MI, ki),
                (P7P_II, ki),
            ];
            for &(tg, kb) in &trans_specs {
                let mut tmp = [0.0f32; 4];
                for z in 0..4 {
                    let node = kb + z * nqf;
                    tmp[z] = if node < m {
                        gm.tsc(node, tg).exp()
                    } else {
                        0.0
                    };
                }
                tfv[j] = tmp;
                j += 1;
            }
        }
        // DD transitions at the end
        for qi in 0..nqf {
            let ki = qi + 1;
            let mut tmp = [0.0f32; 4];
            for z in 0..4 {
                let node = ki + z * nqf;
                tmp[z] = if node < m {
                    gm.tsc(node, P7P_DD).exp()
                } else {
                    0.0
                };
            }
            tfv[j] = tmp;
            j += 1;
        }

        // Special state float scores
        let mut xf = [[0.0f32; P7O_NXTRANS]; P7O_NXSTATES];
        xf[P7O_E][P7O_LOOP] = gm.xsc[P7P_E][P7P_LOOP].exp();
        xf[P7O_E][P7O_MOVE] = gm.xsc[P7P_E][P7P_MOVE].exp();
        xf[P7O_N][P7O_LOOP] = gm.xsc[P7P_N][P7P_LOOP].exp();
        xf[P7O_N][P7O_MOVE] = gm.xsc[P7P_N][P7P_MOVE].exp();
        xf[P7O_C][P7O_LOOP] = gm.xsc[P7P_C][P7P_LOOP].exp();
        xf[P7O_C][P7O_MOVE] = gm.xsc[P7P_C][P7P_MOVE].exp();
        xf[P7O_J][P7O_LOOP] = gm.xsc[P7P_J][P7P_LOOP].exp();
        xf[P7O_J][P7O_MOVE] = gm.xsc[P7P_J][P7P_MOVE].exp();

        OProfile {
            rbv,
            tbm_b,
            tec_b,
            tjb_b,
            scale_b,
            base_b,
            bias_b,
            rwv,
            twv,
            xw,
            scale_w,
            base_w,
            ddbound_w,
            ncj_roundoff: 0.0,
            rfv,
            tfv,
            xf,
            m,
            l: gm.l,
            nj: gm.nj,
            mode: gm.mode,
            evparam: gm.evparam,
            cutoff: gm.cutoff,
            compo: gm.compo,
            name: gm.name.clone(),
            abc_k: k,
            abc_kp: kp,
        }
    }

    /// Reconfigure for a new target sequence length.
    pub fn reconfig_length(&mut self, l: i32) {
        self.l = l;
        self.tjb_b = unbiased_byteify(
            self.scale_b,
            (3.0_f32 / (l as f32 + 3.0)).ln(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use std::path::Path;

    #[test]
    fn test_oprofile_convert() {
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
        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);

        let om = OProfile::convert(&gm);
        assert_eq!(om.m, 20);
        assert!(om.scale_b > 4.0 && om.scale_b < 5.0);
        assert_eq!(om.base_b, 190);
        assert!(om.bias_b > 0);
    }

    #[test]
    fn test_nqb() {
        assert_eq!(nqb(20), 2);
        assert_eq!(nqb(16), 2);
        assert_eq!(nqb(17), 2);
        assert_eq!(nqb(32), 2);
        assert_eq!(nqb(33), 3);
        assert_eq!(nqb(100), 7);
    }
}
