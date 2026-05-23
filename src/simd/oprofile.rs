//! P7_OPROFILE - SSE2-optimized scoring profile.
//! Direct port of impl_sse/p7_oprofile.c.

use crate::profile::*;

/// Striped segment length for byte-precision (MSV) vectors: ceil(M/16), min 2.
/// Mirrors C macro `p7O_NQB`.
pub fn nqb(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 16) + 1)
}

/// Striped segment length for word-precision (Viterbi) vectors: ceil(M/8), min 2.
/// Mirrors C macro `p7O_NQW`.
pub fn nqw(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 8) + 1)
}

/// Striped segment length for float-precision (Forward) vectors: ceil(M/4), min 2.
/// Mirrors C macro `p7O_NQF`.
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

/// Emit a tracehash record of the Forward-space rfv/tfv/xf vectors of `om`,
/// for cross-implementation diff debugging. Gated on `tracehash` feature.
#[cfg(feature = "tracehash")]
fn trace_oprofile_fwd_vectors(om: &OProfile) {
    for x in 0..om.abc_kp {
        for (q, lanes) in om.rfv[x].iter().enumerate() {
            let mut th = tracehash::th_call!("oprofile_rfv_bits");
            th.input_usize(om.m);
            th.input_usize(om.abc_kp);
            th.input_usize(x);
            th.input_usize(q);
            for lane in lanes {
                th.output_u64(lane.to_bits() as u64);
            }
            th.finish();
        }
    }

    for (idx, lanes) in om.tfv.iter().enumerate() {
        let mut th = tracehash::th_call!("oprofile_tfv_bits");
        th.input_usize(om.m);
        th.input_usize(idx);
        for lane in lanes {
            th.output_u64(lane.to_bits() as u64);
        }
        th.finish();
    }

    for state in 0..P7O_NXSTATES {
        for trans in 0..P7O_NXTRANS {
            let mut th = tracehash::th_call!("oprofile_xf_bits");
            th.input_usize(om.m);
            th.input_usize(state);
            th.input_usize(trans);
            th.output_u64(om.xf[state][trans].to_bits() as u64);
            th.finish();
        }
    }
}

/// Emit a tracehash record of the raw log-space tfv lanes before exp() lifting.
#[cfg(feature = "tracehash")]
fn trace_oprofile_tfv_source(m: usize, idx: usize, lanes: &[f32; 4]) {
    let mut th = tracehash::th_call!("oprofile_tfv_source_bits");
    th.input_usize(m);
    th.input_usize(idx);
    for lane in lanes {
        th.output_u64(lane.to_bits() as u64);
    }
    th.finish();
}

/// 16-byte aligned wrapper around `[f32; 4]` for use with SSE `_mm_load_ps`.
#[cfg(target_arch = "x86_64")]
#[repr(align(16))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF32x4(pub [f32; 4]);

#[cfg(target_arch = "x86_64")]
impl AlignedF32x4 {
    /// Wrap a `[f32; 4]` into an aligned lane container.
    #[inline]
    pub fn from_array(v: [f32; 4]) -> Self {
        Self(v)
    }

    /// Return a `*const f32` to the first lane (16-byte aligned).
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }
}

/// Optimized profile for SSE2 SIMD operations.
#[derive(Debug, Clone)]
pub struct OProfile {
    // === MSV byte-precision fields ===
    /// MSV byte-precision match scores: `rbv[residue_code][q_vector][16 bytes]`
    pub rbv: Vec<Vec<[u8; 16]>>,
    /// SSV signed-byte match scores, with extra wrapped vectors for band code.
    pub sbv: Vec<Vec<[u8; 16]>>,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,

    // === Viterbi word-precision fields ===
    /// Word-precision match emission scores: `rwv[residue_code][q_vector][8 words]`
    pub rwv: Vec<Vec<[i16; 8]>>,
    /// Word-precision transition scores: twv[j][8 words]
    /// Layout: for each q, 7 transitions (BM,MM,IM,DM,MD,MI,II), then DD at end
    pub twv: Vec<[i16; 8]>,
    /// Special state word scores: `xw[state][transition]`
    pub xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub ncj_roundoff: f32,

    // === Forward/Backward float-precision fields ===
    /// Float-precision emission scores (probability ratios): `rfv[residue][q_vector][4 floats]`
    pub rfv: Vec<Vec<[f32; 4]>>,
    /// Float-precision transition scores: tfv[j][4 floats]
    pub tfv: Vec<[f32; 4]>,
    #[cfg(target_arch = "x86_64")]
    pub rfv_a: Vec<Vec<AlignedF32x4>>,
    #[cfg(target_arch = "x86_64")]
    pub tfv_a: Vec<AlignedF32x4>,
    /// Special state float scores: `xf[state][transition]`
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

/// Convert a float log-odds score to a biased MSV byte (`u8`).
///
/// Matches C `biased_byteify`: `result = (uint8_t)(-round(scale_b * sc)) + bias_b`.
/// For positive `sc` the negated rounded value is negative; unsigned wrapping
/// then yields the correct offset byte. Saturates to 255.
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

/// Convert a float log-odds score to a Viterbi-precision signed word (`i16`).
/// Mirrors C `wordify`: rounds `scale_w * sc` and saturates to `[-32768, 32767]`.
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

/// Convert a float log-odds score to an unbiased MSV byte cost.
/// Mirrors C `unbiased_byteify`: negates and rounds, saturating to 255.
fn unbiased_byteify(scale_b: f32, sc: f32) -> u8 {
    let negated = -1.0 * (scale_b * sc).round();
    if negated > 255.0 {
        255
    } else {
        negated as i32 as u8
    }
}

/// Vectorized 4-lane `expf` port of Easel's `esl_sse_expf`, used to lift log
/// scores into probability space when building the Forward/Backward profile.
/// Uses the Cephes minimax polynomial; falls back to `f32::INFINITY` / `0.0`
/// on over/underflow.
///
/// # Safety
/// Requires SSE2 to be available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn esl_sse_expf4(x: [f32; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;

    const CEPHES_P0: f32 = 1.9875691500e-4;
    const CEPHES_P1: f32 = 1.3981999507e-3;
    const CEPHES_P2: f32 = 8.3334519073e-3;
    const CEPHES_P3: f32 = 4.1665795894e-2;
    const CEPHES_P4: f32 = 1.6666665459e-1;
    const CEPHES_P5: f32 = 5.0000001201e-1;
    const C0: f32 = 0.693359375;
    const C1: f32 = -2.12194440e-4;
    const LOG2R: f32 = 1.44269504088896341_f32;
    const MAXLOGF: f32 = 88.3762626647949;
    const MINLOGF: f32 = -88.3762626647949;

    let mut xv = _mm_loadu_ps(x.as_ptr());
    let maxmask = _mm_cmpgt_ps(xv, _mm_set1_ps(MAXLOGF));
    let minmask = _mm_cmple_ps(xv, _mm_set1_ps(MINLOGF));

    let mut fx = _mm_mul_ps(xv, _mm_set1_ps(LOG2R));
    fx = _mm_add_ps(fx, _mm_set1_ps(0.5));

    let mut k = _mm_cvttps_epi32(fx);
    let tmp = _mm_cvtepi32_ps(k);
    let mut mask = _mm_cmpgt_ps(tmp, fx);
    mask = _mm_and_ps(mask, _mm_set1_ps(1.0));
    fx = _mm_sub_ps(tmp, mask);
    k = _mm_cvttps_epi32(fx);

    let tmp = _mm_mul_ps(fx, _mm_set1_ps(C0));
    let zc = _mm_mul_ps(fx, _mm_set1_ps(C1));
    xv = _mm_sub_ps(xv, tmp);
    xv = _mm_sub_ps(xv, zc);
    let z = _mm_mul_ps(xv, xv);

    let mut y = _mm_set1_ps(CEPHES_P0);
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P1));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P2));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P3));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P4));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P5));
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(1.0));

    k = _mm_add_epi32(k, _mm_set1_epi32(127));
    k = _mm_slli_epi32(k, 23);
    fx = _mm_castsi128_ps(k);
    y = _mm_mul_ps(y, fx);

    let infv = _mm_set1_ps(f32::INFINITY);
    y = _mm_or_ps(_mm_and_ps(maxmask, infv), _mm_andnot_ps(maxmask, y));
    y = _mm_andnot_ps(minmask, y);

    let mut out = [0.0_f32; 4];
    _mm_storeu_ps(out.as_mut_ptr(), y);
    out
}

/// Scalar fallback for `esl_sse_expf4` on non-x86 builds.
#[cfg(not(target_arch = "x86_64"))]
fn esl_sse_expf4(x: [f32; 4]) -> [f32; 4] {
    x.map(esl_sse_expf_lane)
}

/// Scalar fallback for one lane of Easel's `esl_sse_expf()`. Uses the same
/// Cephes minimax polynomial as the SSE version for bitwise parity.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn esl_sse_expf_lane(mut x: f32) -> f32 {
    const CEPHES_P0: f32 = 1.9875691500e-4;
    const CEPHES_P1: f32 = 1.3981999507e-3;
    const CEPHES_P2: f32 = 8.3334519073e-3;
    const CEPHES_P3: f32 = 4.1665795894e-2;
    const CEPHES_P4: f32 = 1.6666665459e-1;
    const CEPHES_P5: f32 = 5.0000001201e-1;
    const C0: f32 = 0.693359375;
    const C1: f32 = -2.12194440e-4;
    const LOG2R: f32 = 1.44269504088896341_f32;
    const MAXLOGF: f32 = 88.3762626647949;
    const MINLOGF: f32 = -88.3762626647949;

    if x > MAXLOGF {
        return f32::INFINITY;
    }
    if x <= MINLOGF {
        return 0.0;
    }

    let mut fx = x * LOG2R;
    fx += 0.5;
    let mut k = fx as i32;
    let tmp = k as f32;
    if tmp > fx {
        fx = tmp - 1.0;
        k -= 1;
    } else {
        fx = tmp;
    }

    let tmp = fx * C0;
    let z = fx * C1;
    x -= tmp;
    x -= z;
    let z = x * x;

    let mut y = CEPHES_P0;
    y *= x;
    y += CEPHES_P1;
    y *= x;
    y += CEPHES_P2;
    y *= x;
    y += CEPHES_P3;
    y *= x;
    y += CEPHES_P4;
    y *= x;
    y += CEPHES_P5;
    y *= z;
    y += x;
    y += 1.0;

    let pow2 = f32::from_bits(((k + 127) as u32) << 23);
    y * pow2
}

impl OProfile {
    /// Convert a standard `Profile` into the SSE-striped optimized profile.
    ///
    /// Ports C `p7_oprofile_Convert()`: runs the MSV (byte), Viterbi (word),
    /// and Forward/Backward (float) striping conversions in one pass.
    /// Requires `gm.m <= allocM` and matching alphabet. Sets `mode`, `L`, `M`,
    /// `nj`, `evparam`, `cutoff`, `compo`, and copies the model name.
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

        let extra_sb = 17;
        let mut sbv = vec![vec![[0u8; 16]; nq + extra_sb]; kp];
        for x in 0..kp {
            for q in 0..(nq + extra_sb) {
                let src = rbv[x][q % nq];
                let mut tmp = [0u8; 16];
                for z in 0..16 {
                    tmp[z] = (bias_b.wrapping_add(127).saturating_sub(src[z])) ^ 127;
                }
                sbv[x][q] = tmp;
            }
        }

        // Transition costs
        let tbm_b = unbiased_byteify(scale_b, (2.0_f32 / (m as f32 * (m as f32 + 1.0))).ln());
        let tec_b = unbiased_byteify(scale_b, 0.5_f32.ln());
        let tjb_b = unbiased_byteify(scale_b, (3.0_f32 / (gm.l as f32 + 3.0)).ln());

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
                (P7P_BM, ki.wrapping_sub(1), 0), // BM: off-by-one, starts from k=0
                (P7P_MM, ki.wrapping_sub(1), 0), // MM: rotated by -1
                (P7P_IM, ki.wrapping_sub(1), 0), // IM: rotated by -1
                (P7P_DM, ki.wrapping_sub(1), 0), // DM: rotated by -1
                (P7P_MD, ki, 0),                 // MD: straight
                (P7P_MI, ki, 0),                 // MI: straight
                (P7P_II, ki, -1),                // II: maxval=-1 (prevent zero-cost II)
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
                        gm.msc(node, x)
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                #[cfg(target_arch = "x86_64")]
                {
                    rfv[x][qi] = unsafe { esl_sse_expf4(tmp) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    rfv[x][qi] = esl_sse_expf4(tmp);
                }
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
                        gm.tsc(node, tg)
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                #[cfg(target_arch = "x86_64")]
                {
                    #[cfg(feature = "tracehash")]
                    trace_oprofile_tfv_source(m, j, &tmp);
                    tfv[j] = unsafe { esl_sse_expf4(tmp) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    #[cfg(feature = "tracehash")]
                    trace_oprofile_tfv_source(m, j, &tmp);
                    tfv[j] = esl_sse_expf4(tmp);
                }
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
                    gm.tsc(node, P7P_DD)
                } else {
                    f32::NEG_INFINITY
                };
            }
            #[cfg(target_arch = "x86_64")]
            {
                #[cfg(feature = "tracehash")]
                trace_oprofile_tfv_source(m, j, &tmp);
                tfv[j] = unsafe { esl_sse_expf4(tmp) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                #[cfg(feature = "tracehash")]
                trace_oprofile_tfv_source(m, j, &tmp);
                tfv[j] = esl_sse_expf4(tmp);
            }
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

        #[cfg(target_arch = "x86_64")]
        let rfv_a: Vec<Vec<AlignedF32x4>> = rfv
            .iter()
            .map(|rows| rows.iter().copied().map(AlignedF32x4::from_array).collect())
            .collect();
        #[cfg(target_arch = "x86_64")]
        let tfv_a: Vec<AlignedF32x4> = tfv.iter().copied().map(AlignedF32x4::from_array).collect();

        let om = OProfile {
            rbv,
            sbv,
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
            #[cfg(target_arch = "x86_64")]
            rfv_a,
            #[cfg(target_arch = "x86_64")]
            tfv_a,
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
        };

        #[cfg(feature = "tracehash")]
        trace_oprofile_fwd_vectors(&om);

        om
    }

    /// Set the target sequence length of the MSVFilter part of the model.
    ///
    /// Ports `p7_oprofile_ReconfigMSVLength`: recomputes only `tjb_b` for a
    /// new mean target length `l`. The acceleration pipeline (and nhmmer's
    /// long-target SSV scan) uses this to defer reconfiguring the rest of the
    /// length model until after the MSV stage.
    pub fn reconfig_msv_length(&mut self, l: i32) {
        self.tjb_b = unbiased_byteify(self.scale_b, (3.0_f32 / (l as f32 + 3.0)).ln());
    }

    /// Set the target sequence length of the optimized model.
    ///
    /// Ports `p7_oprofile_ReconfigLength` (combines MSV + "rest" length
    /// reconfigs): updates `tjb_b`, N/C/J transition floats `xf` and words
    /// `xw` for a new mean target length `l`. Does not touch the null model
    /// (caller must also `p7_bg_SetLength`). Must be fast — sits in the hot
    /// path, called once per target sequence.
    pub fn reconfig_length(&mut self, l: i32) {
        self.l = l;
        let pmove = (2.0 + self.nj) / (l as f32 + 2.0 + self.nj);
        let ploop = 1.0 - pmove;

        self.tjb_b = unbiased_byteify(self.scale_b, (3.0_f32 / (l as f32 + 3.0)).ln());
        self.xw[P7O_N][P7O_MOVE] = wordify(self.scale_w, pmove.ln());
        self.xw[P7O_C][P7O_MOVE] = wordify(self.scale_w, pmove.ln());
        self.xw[P7O_J][P7O_MOVE] = wordify(self.scale_w, pmove.ln());

        self.xf[P7O_N][P7O_LOOP] = ploop;
        self.xf[P7O_N][P7O_MOVE] = pmove;
        self.xf[P7O_C][P7O_LOOP] = ploop;
        self.xf[P7O_C][P7O_MOVE] = pmove;
        self.xf[P7O_J][P7O_LOOP] = ploop;
        self.xf[P7O_J][P7O_MOVE] = pmove;
    }

    /// Reconfigure the optimized profile into multihit mode for target length `l`.
    ///
    /// Ports `p7_oprofile_ReconfigMultihit`: sets E->{move,loop}=0.5 and
    /// `nj=1`, mirroring the unihit/multihit flip done during domain
    /// definition. Calls `reconfig_length(l)` to refresh the length model.
    /// Avoids rebuilding through generic log-space scores so low-bit parity
    /// with the original is preserved.
    pub fn reconfig_multihit(&mut self, l: i32) {
        self.xf[P7O_E][P7O_MOVE] = 0.5;
        self.xf[P7O_E][P7O_LOOP] = 0.5;
        self.nj = 1.0;

        let e = -std::f32::consts::LN_2;
        self.xw[P7O_E][P7O_MOVE] = wordify(self.scale_w, e);
        self.xw[P7O_E][P7O_LOOP] = wordify(self.scale_w, e);

        self.reconfig_length(l);
    }

    /// Reconfigure the optimized profile into unihit mode for target length `l`.
    ///
    /// Ports `p7_oprofile_ReconfigUnihit`: sets E->move=1, E->loop=0, `nj=0`,
    /// and refreshes the length model via `reconfig_length(l)`. Used by domain
    /// definition to score isolated single-domain envelopes.
    pub fn reconfig_unihit(&mut self, l: i32) {
        self.xf[P7O_E][P7O_MOVE] = 1.0;
        self.xf[P7O_E][P7O_LOOP] = 0.0;
        self.nj = 0.0;

        self.xw[P7O_E][P7O_MOVE] = 0;
        self.xw[P7O_E][P7O_LOOP] = i16::MIN;

        self.reconfig_length(l);
    }

    /// Retrieve Forward (float) residue emission values into a flat array.
    ///
    /// Ports `p7_oprofile_GetFwdEmissionArray`: extracts an implicit 2D
    /// `(M+1) * Kp` table from the striped/interleaved `rfv`, converting back
    /// to emission values weighted by background. Canonical residues:
    /// `arr[k * Kp + x] = rfv[x][q][z] * bg.f[x]`. Degenerate / ambiguity
    /// codes are filled via `esl_abc_FExpectScVec`-style expectation over
    /// canonical residues weighted by `bg.f`.
    pub fn get_fwd_emissions(
        &self,
        bg: &crate::bg::Bg,
        abc: &crate::alphabet::Alphabet,
    ) -> Vec<f32> {
        let m = self.m;
        let kp = self.abc_kp;
        let k = abc.k;
        let nq = nqf(m);
        let cell_cnt = (m + 1) * kp;
        let mut arr = vec![0.0_f32; cell_cnt];

        for x in 0..k {
            for q in 0..nq {
                let lanes = self.rfv[x][q];
                for z in 0..4 {
                    let node = q + 1 + z * nq;
                    let idx = kp * node + x;
                    if idx < cell_cnt {
                        arr[idx] = lanes[z] * bg.f[x];
                    }
                }
            }
        }

        // Degenerate residue expectations: for each model position, fill in the
        // ambiguity columns as the bg.f-weighted mean of the matching canonical
        // values. Mirrors esl_abc_FExpectScVec over raw probabilities.
        for pos in 0..=m {
            let row = &mut arr[pos * kp..(pos + 1) * kp];
            for x in (k + 1)..=kp.saturating_sub(3) {
                if !abc.is_residue(x as u8) {
                    continue;
                }
                let mut numer = 0.0_f32;
                let mut denom = 0.0_f32;
                for i in 0..k {
                    if abc.degen[x][i] {
                        numer += row[i] * bg.f[i];
                        denom += bg.f[i];
                    }
                }
                row[x] = if denom > 0.0 { numer / denom } else { 0.0 };
            }
        }

        arr
    }

    /// Update the Forward/Backward match emissions for a new bg distribution.
    ///
    /// Ports `p7_oprofile_UpdateFwdEmissionScores`: rewrites the striped `rfv`
    /// (and its aligned mirror `rfv_a`) in place using precomputed raw
    /// Forward emissions from `get_fwd_emissions`. Reorders the loops over
    /// `(q, x, z)` to minimize working scratch. Used by nhmmer long_target
    /// per-envelope reparameterization.
    #[cfg(target_arch = "x86_64")]
    pub fn update_fwd_emission_scores(
        &mut self,
        bg: &crate::bg::Bg,
        fwd_emissions: &[f32],
        abc: &crate::alphabet::Alphabet,
    ) {
        let m = self.m;
        let kp = self.abc_kp;
        let k = abc.k;
        let nq = nqf(m);

        // Row in sc_arr: for each of the 4 lanes (z), Kp residue scores.
        let mut sc_arr = vec![0.0_f32; kp * 4];

        for q in 0..nq {
            // Canonical residues — compute log(prob / bg.f)
            for x in 0..k {
                for z in 0..4 {
                    let node = q + 1 + z * nq;
                    sc_arr[z * kp + x] = if node <= m {
                        ((fwd_emissions[kp * node + x] as f64) / (bg.f[x] as f64)).ln() as f32
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                let tmp = [
                    sc_arr[0 * kp + x],
                    sc_arr[1 * kp + x],
                    sc_arr[2 * kp + x],
                    sc_arr[3 * kp + x],
                ];
                let packed = unsafe { esl_sse_expf4(tmp) };
                self.rfv[x][q] = packed;
                self.rfv_a[x][q] = AlignedF32x4::from_array(packed);
            }

            // Gap, nonresidue, missing codes — -infinity.
            for z in 0..4 {
                sc_arr[z * kp + k] = f32::NEG_INFINITY;
                sc_arr[z * kp + kp - 2] = f32::NEG_INFINITY;
                sc_arr[z * kp + kp - 1] = f32::NEG_INFINITY;
            }

            // Ambiguity codes: expectation over canonical codes.
            for z in 0..4 {
                let base = z * kp;
                for x in (k + 1)..=kp.saturating_sub(3) {
                    if !abc.is_residue(x as u8) {
                        continue;
                    }
                    let mut numer = 0.0_f32;
                    let mut denom = 0.0_f32;
                    for i in 0..k {
                        if abc.degen[x][i] {
                            numer += sc_arr[base + i] * bg.f[i];
                            denom += bg.f[i];
                        }
                    }
                    sc_arr[base + x] = if denom > 0.0 { numer / denom } else { 0.0 };
                }
            }

            for x in k..kp {
                let tmp = [
                    sc_arr[0 * kp + x],
                    sc_arr[1 * kp + x],
                    sc_arr[2 * kp + x],
                    sc_arr[3 * kp + x],
                ];
                let packed = unsafe { esl_sse_expf4(tmp) };
                self.rfv[x][q] = packed;
                self.rfv_a[x][q] = AlignedF32x4::from_array(packed);
            }
        }
    }

    /// Recover a profile transition log-odds score from the striped Viterbi `twv`.
    ///
    /// Rust-only helper used by AVX2/NEON restripers when only the SSE-built
    /// `OProfile` is available. Maps `(node, tsc_type)` back to the striped
    /// layout and de-scales by `scale_w`; returns `NEG_INFINITY` out of range.
    pub fn tsc_at(&self, node: usize, tsc_type: usize) -> f32 {
        // Look up from the generic profile's transition score
        // This requires the profile data stored in word-precision twv
        // For restriping, we approximate from the word value
        let nq = nqw(self.m);
        let q = (node) % nq;
        let z = (node) / nq;
        if z < 8 && q < self.twv.len() / 8 {
            // Map back: twv layout is [q*7+t] for t=0..6, then DD at 7*nq+q
            let idx = if tsc_type == P7P_DD {
                7 * nq + q
            } else {
                q * 7 + tsc_type
            };
            if idx < self.twv.len() && z < 8 {
                return self.twv[idx][z] as f32 / self.scale_w;
            }
        }
        f32::NEG_INFINITY
    }
}

/// Public re-export of `wordify` for use by AVX2/NEON restriping.
/// Mirrors C `wordify`: rounds `scale_w * sc` and saturates to i16 range.
pub fn wordify_pub(scale_w: f32, sc: f32) -> i16 {
    let sc = (scale_w * sc).round();
    if sc >= 32767.0 {
        32767
    } else if sc <= -32768.0 {
        -32768
    } else {
        sc as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use std::path::Path;

    /// Smoke test: build an OProfile from a real test HMM and check basic invariants.
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

    /// Validate the striped segment-length macro for byte-precision vectors.
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
