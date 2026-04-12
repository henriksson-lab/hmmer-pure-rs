//! AVX2-optimized Forward parser (8x float vectors).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;

/// Number of AVX2 float vectors: ceil(M/8), min 2.
pub fn nqf_avx2(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 8) + 1)
}

/// AVX2 OProfile for Forward float scores.
pub struct OProfileAvx2Fwd {
    pub rfv: Vec<Vec<[f32; 8]>>,
    pub tfv: Vec<[f32; 8]>,
    pub xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],
    pub m: usize,
    pub abc_kp: usize,
}

impl OProfileAvx2Fwd {
    pub fn from_oprofile(om: &OProfile) -> Self {
        let m = om.m;
        let nq = nqf_avx2(m);
        let kp = om.abc_kp;
        let nq_sse = nqf(m);

        let mut rfv = vec![vec![[0.0f32; 8]; nq]; kp];
        for x in 0..kp {
            for q in 0..nq {
                let mut tmp = [0.0f32; 8];
                for z in 0..8 {
                    let node = q + 1 + z * nq;
                    if node <= m {
                        let sse_q = (node - 1) % nq_sse;
                        let sse_z = (node - 1) / nq_sse;
                        if sse_z < 4 && sse_q < om.rfv[x].len() {
                            tmp[z] = om.rfv[x][sse_q][sse_z];
                        }
                    }
                }
                rfv[x][q] = tmp;
            }
        }

        let mut tfv = vec![[0.0f32; 8]; 8 * nq];
        let mut j = 0;
        for qi in 0..nq {
            let ki = qi + 1;
            let specs: [(usize, usize); 7] = [
                (crate::profile::P7P_BM, ki.wrapping_sub(1)),
                (crate::profile::P7P_MM, ki.wrapping_sub(1)),
                (crate::profile::P7P_IM, ki.wrapping_sub(1)),
                (crate::profile::P7P_DM, ki.wrapping_sub(1)),
                (crate::profile::P7P_MD, ki),
                (crate::profile::P7P_MI, ki),
                (crate::profile::P7P_II, ki),
            ];
            for &(tg, kb) in &specs {
                let mut tmp = [0.0f32; 8];
                for z in 0..8 {
                    let node = kb + z * nq;
                    if node < m {
                        tmp[z] = om.tsc_at(node, tg).exp();
                    }
                }
                tfv[j] = tmp;
                j += 1;
            }
        }
        for qi in 0..nq {
            let ki = qi + 1;
            let mut tmp = [0.0f32; 8];
            for z in 0..8 {
                let node = ki + z * nq;
                if node < m {
                    tmp[z] = om.tsc_at(node, crate::profile::P7P_DD).exp();
                }
            }
            tfv[j] = tmp;
            j += 1;
        }

        OProfileAvx2Fwd {
            rfv,
            tfv,
            xf: om.xf,
            m,
            abc_kp: kp,
        }
    }
}

/// AVX2 Forward parser.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn avx2_forward_parser(dsq: &[Dsq], l: usize, om: &OProfileAvx2Fwd) -> f32 {
    let q_count = nqf_avx2(om.m);
    let nscells = 3;
    let mut dp: Vec<__m256> = vec![_mm256_setzero_ps(); q_count * nscells];
    let zerov = _mm256_setzero_ps();

    macro_rules! mmo { ($q:expr) => { dp[$q * nscells + 0] }; }
    macro_rules! dmo { ($q:expr) => { dp[$q * nscells + 1] }; }
    macro_rules! imo { ($q:expr) => { dp[$q * nscells + 2] }; }

    let mut xe: f32 = 0.0;
    let mut xn: f32 = 1.0;
    let mut xj: f32 = 0.0;
    let mut xb: f32 = om.xf[P7O_N][P7O_MOVE];
    let mut xc: f32 = 0.0;
    let mut totscale: f32 = 0.0;

    for i in 1..=l {
        let xi = dsq[i] as usize;
        if xi >= om.abc_kp { continue; }
        let rsc = &om.rfv[xi];

        let mut dcv = zerov;
        let mut xev = zerov;
        let xbv = _mm256_set1_ps(xb);

        let mut mpv = rightshift_ps_avx2(mmo!(q_count - 1));
        let mut dpv = rightshift_ps_avx2(dmo!(q_count - 1));
        let mut ipv = rightshift_ps_avx2(imo!(q_count - 1));

        let mut tsc_idx = 0;
        for q in 0..q_count {
            let tbm = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tmm = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tim = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tdm = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;

            let mut sv = _mm256_mul_ps(xbv, tbm);
            sv = _mm256_add_ps(sv, _mm256_mul_ps(mpv, tmm));
            sv = _mm256_add_ps(sv, _mm256_mul_ps(ipv, tim));
            sv = _mm256_add_ps(sv, _mm256_mul_ps(dpv, tdm));
            let rsc_v = _mm256_loadu_ps(rsc[q].as_ptr());
            sv = _mm256_mul_ps(sv, rsc_v);
            xev = _mm256_add_ps(xev, sv);

            mpv = mmo!(q);
            dpv = dmo!(q);
            ipv = imo!(q);

            mmo!(q) = sv;
            dmo!(q) = dcv;

            let tmd = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            dcv = _mm256_mul_ps(sv, tmd);

            let tmi = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            let tii = _mm256_loadu_ps(om.tfv[tsc_idx].as_ptr()); tsc_idx += 1;
            imo!(q) = _mm256_add_ps(
                _mm256_mul_ps(mpv, tmi),
                _mm256_mul_ps(ipv, tii),
            );
        }

        // DD paths
        dcv = rightshift_ps_avx2(dcv);
        dmo!(0) = zerov;
        let dd_offset = 7 * q_count;
        for q in 0..q_count {
            dmo!(q) = _mm256_add_ps(dcv, dmo!(q));
            let tdd = _mm256_loadu_ps(om.tfv[dd_offset + q].as_ptr());
            dcv = _mm256_mul_ps(dmo!(q), tdd);
        }
        for _ in 1..4 {
            dcv = rightshift_ps_avx2(dcv);
            for q in 0..q_count {
                dmo!(q) = _mm256_add_ps(dcv, dmo!(q));
                let tdd = _mm256_loadu_ps(om.tfv[dd_offset + q].as_ptr());
                dcv = _mm256_mul_ps(dcv, tdd);
            }
        }

        for q in 0..q_count { xev = _mm256_add_ps(dmo!(q), xev); }

        // Horizontal sum
        let hi = _mm256_extractf128_ps::<1>(xev);
        let lo = _mm256_castps256_ps128(xev);
        let sum4 = _mm_add_ps(hi, lo);
        let sum4 = _mm_add_ps(sum4, _mm_shuffle_ps::<0x4E>(sum4, sum4));
        let sum4 = _mm_add_ps(sum4, _mm_shuffle_ps::<0xB1>(sum4, sum4));
        _mm_store_ss(&mut xe, sum4);

        xn *= om.xf[P7O_N][P7O_LOOP];
        xc = xc * om.xf[P7O_C][P7O_LOOP] + xe * om.xf[P7O_E][P7O_MOVE];
        xj = xj * om.xf[P7O_J][P7O_LOOP] + xe * om.xf[P7O_E][P7O_LOOP];
        xb = xj * om.xf[P7O_J][P7O_MOVE] + xn * om.xf[P7O_N][P7O_MOVE];

        if xe > 1.0e4 {
            let scale = 1.0 / xe;
            xn *= scale; xc *= scale; xj *= scale; xb *= scale;
            let sv = _mm256_set1_ps(scale);
            for q in 0..q_count {
                mmo!(q) = _mm256_mul_ps(mmo!(q), sv);
                dmo!(q) = _mm256_mul_ps(dmo!(q), sv);
                imo!(q) = _mm256_mul_ps(imo!(q), sv);
            }
            totscale += xe.ln();
            xe = 1.0;
        }
    }

    if xc.is_nan() || (l > 0 && xc == 0.0) || xc.is_infinite() {
        return f32::NEG_INFINITY;
    }
    totscale + (xc * om.xf[P7O_C][P7O_MOVE]).ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn rightshift_ps_avx2(v: __m256) -> __m256 {
    // Cross-lane shift by 4 bytes (1 float)
    let perm = _mm256_permute2f128_ps(v, v, 0x08); // lane0=zero, lane1=old_lane0
    let shifted = _mm256_castsi256_ps(_mm256_alignr_epi8::<4>(
        _mm256_castps_si256(v),
        _mm256_castps_si256(perm),
    ));
    // Zero the first float
    _mm256_blend_ps::<0x01>( shifted, _mm256_setzero_ps())
}
