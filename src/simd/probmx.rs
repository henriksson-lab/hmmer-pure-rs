//! Probability-space DP matrix for SIMD Forward/Backward results.
//! Stores per-position special states (E/N/J/B/C) and cumulative log-scale.
//! Used for domain decoding without needing full per-M-state DP rows.

use crate::dp::gmx::*;

/// Special state indices in xmx array.
pub const PXE: usize = 0;
pub const PXN: usize = 1;
pub const PXJ: usize = 2;
pub const PXB: usize = 3;
pub const PXC: usize = 4;
const NXCELLS: usize = 5;
const DP_CELLS_PER_K: usize = 3; // M, I, D

/// Probability-space DP matrix for SIMD Forward/Backward.
/// Stores per-position special states + cumulative scale (always).
/// Optionally stores full per-M-state DP rows (for posterior decoding / null2).
pub struct ProbMx {
    pub m: usize,
    pub l: usize,
    /// Special states: xmx[(l+1) * 5], indexed as xmx[i * 5 + state]
    pub xmx: Vec<f32>,
    /// Cumulative log-scale per position (f64 for precision)
    pub scale: Vec<f64>,
    /// Per-row scale factor, matching P7_OMX xmx[p7X_SCALE].
    pub row_scale: Vec<f32>,
    /// Full DP rows (optional): dp[(l+1) * (m+1) * 3]
    /// dp[i * row_width + k * 3 + s] for M(0)/I(1)/D(2) at position i, node k.
    pub dp: Vec<f32>,
    /// Full DP rows in HMMER's striped SIMD layout:
    /// striped_dp[i * striped_row_width + q * 12 + s * 4 + lane].
    pub striped_dp: Vec<f32>,
    row_width: usize,
    striped_row_width: usize,
    q: usize,
    pub has_dp: bool,
    pub has_own_scales: bool,
}

impl ProbMx {
    /// Create parser-mode ProbMx (specials + scale only, no DP rows).
    pub fn new(l: usize) -> Self {
        ProbMx {
            m: 0,
            l,
            xmx: vec![0.0; (l + 1) * NXCELLS],
            scale: vec![0.0; l + 1],
            row_scale: vec![1.0; l + 1],
            dp: Vec::new(),
            striped_dp: Vec::new(),
            row_width: 0,
            striped_row_width: 0,
            q: 0,
            has_dp: false,
            has_own_scales: false,
        }
    }

    /// Create full-matrix ProbMx (specials + scale + DP rows).
    pub fn new_full(m: usize, l: usize) -> Self {
        let row_width = (m + 1) * DP_CELLS_PER_K;
        let q = m.div_ceil(4);
        let striped_row_width = q * DP_CELLS_PER_K * 4;
        ProbMx {
            m,
            l,
            xmx: vec![0.0; (l + 1) * NXCELLS],
            scale: vec![0.0; l + 1],
            row_scale: vec![1.0; l + 1],
            dp: Vec::new(),
            striped_dp: vec![0.0; (l + 1) * striped_row_width],
            row_width,
            striped_row_width,
            q,
            has_dp: true,
            has_own_scales: false,
        }
    }

    /// Reuse this matrix as a full parser DP matrix.
    pub fn resize_full(&mut self, m: usize, l: usize) {
        let row_width = (m + 1) * DP_CELLS_PER_K;
        let q = m.div_ceil(4);
        let striped_row_width = q * DP_CELLS_PER_K * 4;

        self.m = m;
        self.l = l;
        self.row_width = row_width;
        self.striped_row_width = striped_row_width;
        self.q = q;
        self.has_dp = true;
        self.has_own_scales = false;

        self.xmx.resize((l + 1) * NXCELLS, 0.0);
        self.scale.resize(l + 1, 0.0);
        self.row_scale.resize(l + 1, 1.0);
        self.dp.clear();
        self.striped_dp.resize((l + 1) * striped_row_width, 0.0);
    }

    #[inline]
    pub fn xmx(&self, i: usize, s: usize) -> f32 {
        self.xmx[i * NXCELLS + s]
    }

    #[inline]
    pub fn set_xmx(&mut self, i: usize, s: usize, val: f32) {
        self.xmx[i * NXCELLS + s] = val;
    }

    /// Get M-state posterior at position i, node k.
    #[inline]
    pub fn mmx(&self, i: usize, k: usize) -> f32 {
        if !self.striped_dp.is_empty() {
            let qi = (k - 1) % self.q;
            let lane = (k - 1) / self.q;
            self.striped_dp[i * self.striped_row_width + qi * DP_CELLS_PER_K * 4 + lane]
        } else {
            self.dp[i * self.row_width + k * DP_CELLS_PER_K]
        }
    }

    /// Get I-state posterior at position i, node k.
    #[inline]
    pub fn imx(&self, i: usize, k: usize) -> f32 {
        if !self.striped_dp.is_empty() {
            let qi = (k - 1) % self.q;
            let lane = (k - 1) / self.q;
            self.striped_dp[i * self.striped_row_width + qi * DP_CELLS_PER_K * 4 + 8 + lane]
        } else {
            self.dp[i * self.row_width + k * DP_CELLS_PER_K + 1]
        }
    }

    /// Get D-state value at position i, node k.
    #[inline]
    pub fn dmx(&self, i: usize, k: usize) -> f32 {
        if !self.striped_dp.is_empty() {
            let qi = (k - 1) % self.q;
            let lane = (k - 1) / self.q;
            self.striped_dp[i * self.striped_row_width + qi * DP_CELLS_PER_K * 4 + 4 + lane]
        } else {
            self.dp[i * self.row_width + k * DP_CELLS_PER_K + 2]
        }
    }

    /// Set M-state value.
    #[inline]
    pub fn set_mmx(&mut self, i: usize, k: usize, val: f32) {
        if !self.striped_dp.is_empty() && k > 0 {
            let qi = (k - 1) % self.q;
            let lane = (k - 1) / self.q;
            self.striped_dp[i * self.striped_row_width + qi * DP_CELLS_PER_K * 4 + lane] = val;
        } else if !self.dp.is_empty() {
            let idx = i * self.row_width + k * DP_CELLS_PER_K;
            self.dp[idx] = val;
        }
    }

    /// Set I-state value.
    #[inline]
    pub fn set_imx(&mut self, i: usize, k: usize, val: f32) {
        if !self.striped_dp.is_empty() && k > 0 {
            let qi = (k - 1) % self.q;
            let lane = (k - 1) / self.q;
            self.striped_dp[i * self.striped_row_width + qi * DP_CELLS_PER_K * 4 + 8 + lane] = val;
        } else if !self.dp.is_empty() {
            let idx = i * self.row_width + k * DP_CELLS_PER_K + 1;
            self.dp[idx] = val;
        }
    }

    /// Write a SIMD DP row (de-striped from __m128 vectors) into position i.
    /// `dp_simd` is a slice of q*3 __m128 vectors (M/D/I interleaved per stripe).
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn write_simd_row(
        &mut self,
        dp_simd: &[std::arch::x86_64::__m128],
        q: usize,
        _m: usize,
        i: usize,
    ) {
        use std::arch::x86_64::*;
        let row_base = i * self.striped_row_width;
        for qi in 0..q {
            let off = row_base + qi * DP_CELLS_PER_K * 4;
            _mm_storeu_ps(self.striped_dp.as_mut_ptr().add(off), dp_simd[qi * 3]);
            _mm_storeu_ps(
                self.striped_dp.as_mut_ptr().add(off + 4),
                dp_simd[qi * 3 + 1],
            );
            _mm_storeu_ps(
                self.striped_dp.as_mut_ptr().add(off + 8),
                dp_simd[qi * 3 + 2],
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn zero_simd_row(&mut self, i: usize) {
        let row_base = i * self.striped_row_width;
        let row_end = row_base + self.striped_row_width;
        self.striped_dp[row_base..row_end].fill(0.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub unsafe fn striped_row_ptr(&mut self, i: usize) -> *mut f32 {
        self.striped_dp.as_mut_ptr().add(i * self.striped_row_width)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn striped_row_width(&self) -> usize {
        self.striped_row_width
    }

    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn q_count(&self) -> usize {
        self.q
    }

    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_sum(&self, i: usize, state: usize) -> f32 {
        let mut sum = 0.0_f32;
        let row_base = i * self.striped_row_width;
        for qi in 0..self.q {
            let off = row_base + qi * DP_CELLS_PER_K * 4 + state * 4;
            for lane in 0..4 {
                let k = qi + 1 + lane * self.q;
                if k <= self.m {
                    sum += self.striped_dp[off + lane];
                }
            }
        }
        sum
    }

    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_sum_all_lanes(&self, i: usize, state: usize) -> f32 {
        let mut sum = 0.0_f32;
        let row_base = i * self.striped_row_width;
        for qi in 0..self.q {
            let off = row_base + qi * DP_CELLS_PER_K * 4 + state * 4;
            for lane in 0..4 {
                sum += self.striped_dp[off + lane];
            }
        }
        sum
    }

    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_vector(&self, i: usize, state: usize, qi: usize) -> [f32; 4] {
        let off = i * self.striped_row_width + qi * DP_CELLS_PER_K * 4 + state * 4;
        [
            self.striped_dp[off],
            self.striped_dp[off + 1],
            self.striped_dp[off + 2],
            self.striped_dp[off + 3],
        ]
    }

    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_q_range_sum(
        &self,
        i: usize,
        state: usize,
        q_start: usize,
        q_end: usize,
    ) -> f32 {
        let mut sum = 0.0_f32;
        let row_base = i * self.striped_row_width;
        let q_end = q_end.min(self.q);
        for qi in q_start.min(self.q)..q_end {
            let off = row_base + qi * DP_CELLS_PER_K * 4 + state * 4;
            for lane in 0..4 {
                let k = qi + 1 + lane * self.q;
                if k <= self.m {
                    sum += self.striped_dp[off + lane];
                }
            }
        }
        sum
    }
}

/// Domain decoding from probability-space Forward/Backward matrices.
/// Port of p_domain_decoding() — computes btot, etot, mocc using the
/// per-position specials and cumulative scales from parser-mode SIMD.
///
pub fn p_domain_decoding(
    fwd: &ProbMx,
    bck: &ProbMx,
    l: usize,
    njc_loop: [f32; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_loop = njc_loop[0];
    let j_loop = njc_loop[1];
    let c_loop = njc_loop[2];

    let bck_n0 = bck.xmx(0, PXN);
    let mut scaleproduct = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };

    let mut btot = vec![0.0_f32; l + 1];
    let mut etot = vec![0.0_f32; l + 1];
    let mut mocc = vec![0.0_f32; l + 1];

    for i in 1..=l {
        let b = fwd.xmx(i - 1, PXB) * bck.xmx(i - 1, PXB) * fwd.row_scale[i - 1] * scaleproduct;
        btot[i] = btot[i - 1] + b;

        scaleproduct *= fwd.row_scale[i - 1] / bck.row_scale[i - 1];

        let e = fwd.xmx(i, PXE) * bck.xmx(i, PXE) * fwd.row_scale[i] * scaleproduct;
        etot[i] = etot[i - 1] + e;

        let pn = fwd.xmx(i - 1, PXN) * bck.xmx(i, PXN) * n_loop * scaleproduct;
        let pj = fwd.xmx(i - 1, PXJ) * bck.xmx(i, PXJ) * j_loop * scaleproduct;
        let pc = fwd.xmx(i - 1, PXC) * bck.xmx(i, PXC) * c_loop * scaleproduct;
        mocc[i] = 1.0 - pn - pj - pc;
    }

    (btot, etot, mocc)
}

/// Posterior decoding from full-matrix probability-space Forward+Backward.
/// Computes normalized per-state posteriors, then null2 correction.
/// The per-row normalization corrects for any absolute magnitude differences
/// between the SIMD Backward and the true values.
///
/// Port of p_decoding() + null2_correction() from old codebase.
pub fn p_null2_from_pmx(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize, // alphabet K
    dsq: &[u8],
    ienv: usize,
    jenv: usize,
    rsc: &[Vec<f32>], // profile rsc[x][node*P7P_NR + P7P_MSC]
    njc_loop: [f32; 3],
) -> f32 {
    let match_odds = match_odds_from_rsc(rsc, k, m);
    let null2 = p_null2_odds_from_pmx(fwd, bck, m, k, &match_odds, njc_loop);
    let mut correction = 0.0_f32;
    for i in ienv..=jenv {
        let x = dsq[i] as usize;
        if x < null2.len() && null2[x] > 0.0 {
            correction += null2[x].ln();
        }
    }
    correction
}

pub fn match_odds_from_rsc(rsc: &[Vec<f32>], k: usize, m: usize) -> Vec<f32> {
    let p7p_nr = 2usize;
    let width = m + 1;
    let mut odds = vec![0.0_f32; k * width];
    for x in 0..k {
        for node in 1..=m {
            let idx = node * p7p_nr;
            if x < rsc.len() && idx < rsc[x].len() {
                let msc = rsc[x][idx];
                if msc > f32::NEG_INFINITY {
                    odds[x * width + node] = msc.exp();
                }
            }
        }
    }
    odds
}

/// Posterior decoding from full-matrix probability-space Forward+Backward into
/// a generic matrix. This mirrors the SSE `p7_Decoding()` path, not generic
/// `p7_GDecoding()`: rows are not renormalized, and a Backward matrix that was
/// scaled with Forward row scales uses a constant scale product.
pub fn p_decoding_to_gmx(fwd: &ProbMx, bck: &ProbMx, m: usize, njc_loop: [f32; 3], pp: &mut Gmx) {
    if !fwd.striped_dp.is_empty() && !bck.striped_dp.is_empty() {
        p_decoding_to_gmx_striped(fwd, bck, m, njc_loop, pp);
        return;
    }

    let l = fwd.l;
    let bck_n0 = bck.xmx(0, PXN);
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };

    pp.m = m;
    pp.l = l;

    for k in 0..=m {
        pp.set_mmx(0, k, 0.0);
        pp.set_imx(0, k, 0.0);
        pp.set_dmx(0, k, 0.0);
    }
    pp.set_xmx(0, P7G_E, 0.0);
    pp.set_xmx(0, P7G_N, 0.0);
    pp.set_xmx(0, P7G_J, 0.0);
    pp.set_xmx(0, P7G_B, 0.0);
    pp.set_xmx(0, P7G_C, 0.0);

    let mut scaleproduct = inv_total;
    for i in 1..=l {
        let dp_scale = scaleproduct * fwd.row_scale[i];

        pp.set_mmx(i, 0, 0.0);
        pp.set_imx(i, 0, 0.0);
        pp.set_dmx(i, 0, 0.0);

        for node in 1..m {
            let mp = fwd.mmx(i, node) * bck.mmx(i, node) * dp_scale;
            let ip = fwd.imx(i, node) * bck.imx(i, node) * dp_scale;
            pp.set_mmx(i, node, mp);
            pp.set_imx(i, node, ip);
            pp.set_dmx(i, node, 0.0);
        }

        let mp = fwd.mmx(i, m) * bck.mmx(i, m) * dp_scale;
        pp.set_mmx(i, m, mp);
        pp.set_imx(i, m, 0.0);
        pp.set_dmx(i, m, 0.0);

        let pn = fwd.xmx(i - 1, PXN) * bck.xmx(i, PXN) * njc_loop[0] * scaleproduct;
        let pj = fwd.xmx(i - 1, PXJ) * bck.xmx(i, PXJ) * njc_loop[1] * scaleproduct;
        let pc = fwd.xmx(i - 1, PXC) * bck.xmx(i, PXC) * njc_loop[2] * scaleproduct;

        pp.set_xmx(i, P7G_E, 0.0);
        pp.set_xmx(i, P7G_N, pn);
        pp.set_xmx(i, P7G_J, pj);
        pp.set_xmx(i, P7G_B, 0.0);
        pp.set_xmx(i, P7G_C, pc);

        if bck.has_own_scales {
            scaleproduct *= fwd.row_scale[i] / bck.row_scale[i];
        }
    }
}

fn p_decoding_to_gmx_striped(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    njc_loop: [f32; 3],
    pp: &mut Gmx,
) {
    let l = fwd.l;
    let fdp = fwd.striped_dp.as_ptr();
    let bdp = bck.striped_dp.as_ptr();
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let ppdp = pp.dp_mem.as_mut_ptr();
    let ppx = pp.xmx.as_mut_ptr();

    let bck_n0 = unsafe { *bx.add(PXN) };
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let q = fwd.q;
    let pp_w = pp.row_width();
    let pp_stride = pp_w * P7G_NSCELLS;

    pp.m = m;
    pp.l = l;

    unsafe {
        for node in 0..=m {
            let idx = node * P7G_NSCELLS;
            *ppdp.add(idx + P7G_M) = 0.0;
            *ppdp.add(idx + P7G_I) = 0.0;
            *ppdp.add(idx + P7G_D) = 0.0;
        }
        *ppx.add(P7G_E) = 0.0;
        *ppx.add(P7G_N) = 0.0;
        *ppx.add(P7G_J) = 0.0;
        *ppx.add(P7G_B) = 0.0;
        *ppx.add(P7G_C) = 0.0;

        let mut scaleproduct = inv_total;
        for i in 1..=l {
            let dp_scale = scaleproduct * *fwd.row_scale.as_ptr().add(i);
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;
            let prow = i * pp_stride;
            let xrow = i * P7G_NXCELLS;

            *ppdp.add(prow + P7G_M) = 0.0;
            *ppdp.add(prow + P7G_I) = 0.0;
            *ppdp.add(prow + P7G_D) = 0.0;

            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    if node <= m {
                        let pidx = prow + node * P7G_NSCELLS;
                        let mp = *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale;
                        *ppdp.add(pidx + P7G_M) = mp;
                        *ppdp.add(pidx + P7G_D) = 0.0;

                        if node < m {
                            let ip =
                                *fdp.add(fbase + 8 + lane) * *bdp.add(bbase + 8 + lane) * dp_scale;
                            *ppdp.add(pidx + P7G_I) = ip;
                        } else {
                            *ppdp.add(pidx + P7G_I) = 0.0;
                        }
                    }
                }
            }

            let pn = *fx.add((i - 1) * NXCELLS + PXN)
                * *bx.add(i * NXCELLS + PXN)
                * njc_loop[0]
                * scaleproduct;
            let pj = *fx.add((i - 1) * NXCELLS + PXJ)
                * *bx.add(i * NXCELLS + PXJ)
                * njc_loop[1]
                * scaleproduct;
            let pc = *fx.add((i - 1) * NXCELLS + PXC)
                * *bx.add(i * NXCELLS + PXC)
                * njc_loop[2]
                * scaleproduct;

            *ppx.add(xrow + P7G_E) = 0.0;
            *ppx.add(xrow + P7G_N) = pn;
            *ppx.add(xrow + P7G_J) = pj;
            *ppx.add(xrow + P7G_B) = 0.0;
            *ppx.add(xrow + P7G_C) = pc;

            if bck.has_own_scales {
                scaleproduct *=
                    *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
            }
        }
    }
}

/// Compute null2 odds ratios from full-matrix probability-space Forward+Backward.
pub fn p_null2_odds_from_pmx(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize,
    match_odds: &[f32],
    njc_loop: [f32; 3],
) -> Vec<f32> {
    if !fwd.striped_dp.is_empty() && !bck.striped_dp.is_empty() {
        return p_null2_odds_from_striped_pmx(fwd, bck, m, k, match_odds, njc_loop);
    }

    let ienv = 1;
    let jenv = fwd.l;
    let ld = jenv - ienv + 1;
    let bck_n0 = bck.xmx(0, PXN);
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let bck_scale0 = bck.scale[0];

    // Step 1: Compute normalized posteriors and accumulate expected state usage
    let mut exp_m = vec![0.0_f32; m + 1];
    let mut exp_i = vec![0.0_f32; m + 1];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;

    for i in ienv..=jenv {
        let dp_scale = ((fwd.scale[i] + bck.scale[i] - bck_scale0) as f32).exp() * inv_total;

        let mut denom = 0.0_f32;

        for node in 1..=m {
            denom += fwd.mmx(i, node) * bck.mmx(i, node) * dp_scale;
            if node < m {
                denom += fwd.imx(i, node) * bck.imx(i, node) * dp_scale;
            }
        }

        // N/J/C specials (use fwd[i-1] * bck[i] * transition)
        let sp_scale = if i > 0 {
            ((fwd.scale[i - 1] + bck.scale[i] - bck_scale0) as f32).exp() * inv_total
        } else {
            dp_scale
        };
        let pn = fwd.xmx(i.saturating_sub(1), PXN) * bck.xmx(i, PXN) * njc_loop[0] * sp_scale;
        let pj = fwd.xmx(i.saturating_sub(1), PXJ) * bck.xmx(i, PXJ) * njc_loop[1] * sp_scale;
        let pc = fwd.xmx(i.saturating_sub(1), PXC) * bck.xmx(i, PXC) * njc_loop[2] * sp_scale;
        denom += pn + pj + pc;

        // Normalize and accumulate
        if denom > 0.0 {
            let inv = 1.0 / denom;
            for node in 1..=m {
                exp_m[node] += fwd.mmx(i, node) * bck.mmx(i, node) * dp_scale * inv;
                if node < m {
                    exp_i[node] += fwd.imx(i, node) * bck.imx(i, node) * dp_scale * inv;
                }
            }
            exp_n += pn * inv;
            exp_j += pj * inv;
            exp_c += pc * inv;
        }
    }

    // Step 2: Convert to frequencies
    let norm = 1.0 / ld as f32;
    for node in 1..=m {
        exp_m[node] *= norm;
        exp_i[node] *= norm;
    }
    exp_n *= norm;
    exp_c *= norm;
    exp_j *= norm;
    let xfactor = exp_n + exp_c + exp_j;

    // Insert total
    let mut insert_total = 0.0_f32;
    for node in 1..m {
        insert_total += exp_i[node];
    }

    // Step 3: Compute null2 odds ratios
    let mut null2 = vec![1.0_f32; k];
    for x in 0..k {
        let mut val = 0.0_f32;
        for node in 1..=m {
            let odds_idx = x * (m + 1) + node;
            if odds_idx < match_odds.len() {
                val += exp_m[node] * match_odds[odds_idx];
            }
        }
        val += insert_total + xfactor;
        null2[x] = val;
    }
    null2
}

pub fn p_null2_odds_from_pmx_reuse(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize,
    match_odds: &[f32],
    njc_loop: [f32; 3],
    null2: &mut Vec<f32>,
    exp_m: &mut Vec<f32>,
    exp_i: &mut Vec<f32>,
) {
    if !fwd.striped_dp.is_empty() && !bck.striped_dp.is_empty() {
        p_null2_odds_from_striped_pmx_reuse(
            fwd, bck, m, k, match_odds, njc_loop, null2, exp_m, exp_i,
        );
        return;
    }

    *null2 = p_null2_odds_from_pmx(fwd, bck, m, k, match_odds, njc_loop);
}

pub fn p_null2_odds_from_omx_expectation_reuse(
    fwd: &ProbMx,
    bck: &ProbMx,
    k: usize,
    rfv: &[Vec<[f32; 4]>],
    njc_loop: [f32; 3],
    null2: &mut Vec<f32>,
    exp_m: &mut Vec<f32>,
    exp_i: &mut Vec<f32>,
) {
    if fwd.striped_dp.is_empty() || bck.striped_dp.is_empty() {
        return;
    }

    let ld = fwd.l;
    let q = fwd.q;
    let exp_len = q * 4;
    exp_m.resize(exp_len, 0.0);
    exp_i.resize(exp_len, 0.0);
    exp_m.fill(0.0);
    exp_i.fill(0.0);

    let fdp = fwd.striped_dp.as_ptr();
    let bdp = bck.striped_dp.as_ptr();
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let exp_m_ptr = exp_m.as_mut_ptr();
    let exp_i_ptr = exp_i.as_mut_ptr();

    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let mut scaleproduct = {
        let bck_n0 = bck.xmx(0, PXN);
        if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 }
    };

    unsafe {
        for i in 1..=ld {
            let dp_scale = scaleproduct * *fwd.row_scale.as_ptr().add(i);
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;

            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                let ebase = qi * 4;
                for lane in 0..4 {
                    *exp_m_ptr.add(ebase + lane) +=
                        *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale;
                    *exp_i_ptr.add(ebase + lane) +=
                        *fdp.add(fbase + 8 + lane) * *bdp.add(bbase + 8 + lane) * dp_scale;
                }
            }

            exp_n += *fx.add((i - 1) * NXCELLS + PXN)
                * *bx.add(i * NXCELLS + PXN)
                * njc_loop[0]
                * scaleproduct;
            exp_j += *fx.add((i - 1) * NXCELLS + PXJ)
                * *bx.add(i * NXCELLS + PXJ)
                * njc_loop[1]
                * scaleproduct;
            exp_c += *fx.add((i - 1) * NXCELLS + PXC)
                * *bx.add(i * NXCELLS + PXC)
                * njc_loop[2]
                * scaleproduct;

            if bck.has_own_scales {
                scaleproduct *=
                    *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
            }
        }

        let norm = 1.0 / ld as f32;
        for idx in 0..exp_len {
            *exp_m_ptr.add(idx) *= norm;
            *exp_i_ptr.add(idx) *= norm;
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;

        null2.resize(k, 1.0);
        for x in 0..k {
            let mut lanes = [0.0_f32; 4];
            for qi in 0..q {
                let ebase = qi * 4;
                let odds = rfv[x][qi];
                for lane in 0..4 {
                    lanes[lane] += *exp_m_ptr.add(ebase + lane) * odds[lane];
                    lanes[lane] += *exp_i_ptr.add(ebase + lane);
                }
            }
            let h01 = lanes[0] + lanes[1];
            let h23 = lanes[2] + lanes[3];
            null2[x] = (h01 + h23) + xfactor;
        }
    }
}

pub fn p_null2_odds_from_gmx(pp: &Gmx, m: usize, k: usize, match_odds: &[f32]) -> Vec<f32> {
    let mut null2 = Vec::new();
    let mut exp_m = Vec::new();
    let mut exp_i = Vec::new();
    p_null2_odds_from_gmx_reuse(pp, m, k, match_odds, &mut null2, &mut exp_m, &mut exp_i);
    null2
}

pub fn p_null2_odds_from_gmx_reuse(
    pp: &Gmx,
    m: usize,
    k: usize,
    match_odds: &[f32],
    null2: &mut Vec<f32>,
    exp_m: &mut Vec<f32>,
    exp_i: &mut Vec<f32>,
) {
    let ld = pp.l;
    let q = m.div_ceil(4);
    let pp_w = pp.row_width();
    let pp_stride = pp_w * P7G_NSCELLS;
    let ppdp = pp.dp_mem.as_ptr();
    let ppx = pp.xmx.as_ptr();

    let exp_len = q * 4;
    exp_m.resize(exp_len, 0.0);
    exp_i.resize(exp_len, 0.0);
    exp_m.fill(0.0);
    exp_i.fill(0.0);
    let exp_m_ptr = exp_m.as_mut_ptr();
    let exp_i_ptr = exp_i.as_mut_ptr();
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;

    unsafe {
        for i in 1..=ld {
            let prow = i * pp_stride;
            for qi in 0..q {
                let ebase = qi * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    if node <= m {
                        let pidx = prow + node * P7G_NSCELLS;
                        *exp_m_ptr.add(ebase + lane) += *ppdp.add(pidx + P7G_M);
                        if node < m {
                            *exp_i_ptr.add(ebase + lane) += *ppdp.add(pidx + P7G_I);
                        }
                    }
                }
            }
            let xrow = i * P7G_NXCELLS;
            exp_n += *ppx.add(xrow + P7G_N);
            exp_j += *ppx.add(xrow + P7G_J);
            exp_c += *ppx.add(xrow + P7G_C);
        }

        let norm = 1.0 / ld as f32;
        for idx in 0..exp_len {
            *exp_m_ptr.add(idx) *= norm;
            *exp_i_ptr.add(idx) *= norm;
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;
        let mut insert_total = 0.0_f32;
        for idx in 0..exp_len {
            insert_total += *exp_i_ptr.add(idx);
        }

        null2.resize(k, 1.0);
        let null2_ptr = null2.as_mut_ptr();
        let odds_ptr = match_odds.as_ptr();
        let odds_len = match_odds.len();
        for x in 0..k {
            let mut val = 0.0_f32;
            for qi in 0..q {
                let ebase = qi * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    let odds_idx = x * (m + 1) + node;
                    if node <= m && odds_idx < odds_len {
                        val += *exp_m_ptr.add(ebase + lane) * *odds_ptr.add(odds_idx);
                    }
                }
            }
            *null2_ptr.add(x) = val + insert_total + xfactor;
        }
    }
}

fn p_null2_odds_from_striped_pmx(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize,
    match_odds: &[f32],
    njc_loop: [f32; 3],
) -> Vec<f32> {
    let ld = fwd.l;
    let fdp = fwd.striped_dp.as_ptr();
    let bdp = bck.striped_dp.as_ptr();
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let fs = fwd.scale.as_ptr();
    let bs = bck.scale.as_ptr();

    let bck_n0 = unsafe { *bx.add(PXN) };
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let bck_scale0 = unsafe { *bs };
    let q = fwd.q;

    let mut exp_m = vec![0.0_f32; q * 4];
    let mut exp_i = vec![0.0_f32; q * 4];
    let exp_m_ptr = exp_m.as_mut_ptr();
    let exp_i_ptr = exp_i.as_mut_ptr();
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;

    unsafe {
        for i in 1..=ld {
            let dp_scale = ((*fs.add(i) + *bs.add(i) - bck_scale0) as f32).exp() * inv_total;
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;

            let mut denom = 0.0_f32;
            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    if node <= m {
                        denom += *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale;
                        if node < m {
                            denom +=
                                *fdp.add(fbase + 8 + lane) * *bdp.add(bbase + 8 + lane) * dp_scale;
                        }
                    }
                }
            }

            let sp_scale = ((*fs.add(i - 1) + *bs.add(i) - bck_scale0) as f32).exp() * inv_total;
            let pn = *fx.add((i - 1) * NXCELLS + PXN)
                * *bx.add(i * NXCELLS + PXN)
                * njc_loop[0]
                * sp_scale;
            let pj = *fx.add((i - 1) * NXCELLS + PXJ)
                * *bx.add(i * NXCELLS + PXJ)
                * njc_loop[1]
                * sp_scale;
            let pc = *fx.add((i - 1) * NXCELLS + PXC)
                * *bx.add(i * NXCELLS + PXC)
                * njc_loop[2]
                * sp_scale;
            denom += pn + pj + pc;

            if denom > 0.0 {
                let inv = 1.0 / denom;
                for qi in 0..q {
                    let fbase = frow + qi * DP_CELLS_PER_K * 4;
                    let bbase = brow + qi * DP_CELLS_PER_K * 4;
                    let ebase = qi * 4;
                    for lane in 0..4 {
                        let node = 1 + qi + lane * q;
                        if node <= m {
                            *exp_m_ptr.add(ebase + lane) +=
                                *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale * inv;
                            if node < m {
                                *exp_i_ptr.add(ebase + lane) += *fdp.add(fbase + 8 + lane)
                                    * *bdp.add(bbase + 8 + lane)
                                    * dp_scale
                                    * inv;
                            }
                        }
                    }
                }
                exp_n += pn * inv;
                exp_j += pj * inv;
                exp_c += pc * inv;
            }
        }

        let norm = 1.0 / ld as f32;
        let exp_len = q * 4;
        for idx in 0..exp_len {
            *exp_m_ptr.add(idx) *= norm;
            *exp_i_ptr.add(idx) *= norm;
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;
        let mut insert_total = 0.0_f32;
        for idx in 0..exp_len {
            insert_total += *exp_i_ptr.add(idx);
        }

        let mut null2 = vec![1.0_f32; k];
        let null2_ptr = null2.as_mut_ptr();
        let odds_ptr = match_odds.as_ptr();
        let odds_len = match_odds.len();
        for x in 0..k {
            let mut val = 0.0_f32;
            for qi in 0..q {
                let ebase = qi * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    let odds_idx = x * (m + 1) + node;
                    if node <= m && odds_idx < odds_len {
                        val += *exp_m_ptr.add(ebase + lane) * *odds_ptr.add(odds_idx);
                    }
                }
            }
            *null2_ptr.add(x) = val + insert_total + xfactor;
        }
        null2
    }
}

fn p_null2_odds_from_striped_pmx_reuse(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize,
    match_odds: &[f32],
    njc_loop: [f32; 3],
    null2: &mut Vec<f32>,
    exp_m: &mut Vec<f32>,
    exp_i: &mut Vec<f32>,
) {
    let ld = fwd.l;
    let fdp = fwd.striped_dp.as_ptr();
    let bdp = bck.striped_dp.as_ptr();
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let fs = fwd.scale.as_ptr();
    let bs = bck.scale.as_ptr();

    let bck_n0 = unsafe { *bx.add(PXN) };
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let bck_scale0 = unsafe { *bs };
    let q = fwd.q;
    let exp_len = q * 4;

    exp_m.resize(exp_len, 0.0);
    exp_i.resize(exp_len, 0.0);
    exp_m.fill(0.0);
    exp_i.fill(0.0);
    let exp_m_ptr = exp_m.as_mut_ptr();
    let exp_i_ptr = exp_i.as_mut_ptr();
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;

    unsafe {
        for i in 1..=ld {
            let dp_scale = ((*fs.add(i) + *bs.add(i) - bck_scale0) as f32).exp() * inv_total;
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;

            let mut denom = 0.0_f32;
            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    if node <= m {
                        denom += *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale;
                        if node < m {
                            denom +=
                                *fdp.add(fbase + 8 + lane) * *bdp.add(bbase + 8 + lane) * dp_scale;
                        }
                    }
                }
            }

            let sp_scale = ((*fs.add(i - 1) + *bs.add(i) - bck_scale0) as f32).exp() * inv_total;
            let pn = *fx.add((i - 1) * NXCELLS + PXN)
                * *bx.add(i * NXCELLS + PXN)
                * njc_loop[0]
                * sp_scale;
            let pj = *fx.add((i - 1) * NXCELLS + PXJ)
                * *bx.add(i * NXCELLS + PXJ)
                * njc_loop[1]
                * sp_scale;
            let pc = *fx.add((i - 1) * NXCELLS + PXC)
                * *bx.add(i * NXCELLS + PXC)
                * njc_loop[2]
                * sp_scale;
            denom += pn + pj + pc;

            if denom > 0.0 {
                let inv = 1.0 / denom;
                for qi in 0..q {
                    let fbase = frow + qi * DP_CELLS_PER_K * 4;
                    let bbase = brow + qi * DP_CELLS_PER_K * 4;
                    let ebase = qi * 4;
                    for lane in 0..4 {
                        let node = 1 + qi + lane * q;
                        if node <= m {
                            *exp_m_ptr.add(ebase + lane) +=
                                *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale * inv;
                            if node < m {
                                *exp_i_ptr.add(ebase + lane) += *fdp.add(fbase + 8 + lane)
                                    * *bdp.add(bbase + 8 + lane)
                                    * dp_scale
                                    * inv;
                            }
                        }
                    }
                }
                exp_n += pn * inv;
                exp_j += pj * inv;
                exp_c += pc * inv;
            }
        }

        let norm = 1.0 / ld as f32;
        for idx in 0..exp_len {
            *exp_m_ptr.add(idx) *= norm;
            *exp_i_ptr.add(idx) *= norm;
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;
        let mut insert_total = 0.0_f32;
        for idx in 0..exp_len {
            insert_total += *exp_i_ptr.add(idx);
        }

        null2.resize(k, 1.0);
        let null2_ptr = null2.as_mut_ptr();
        let odds_ptr = match_odds.as_ptr();
        let odds_len = match_odds.len();
        for x in 0..k {
            let mut val = 0.0_f32;
            for qi in 0..q {
                let ebase = qi * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    let odds_idx = x * (m + 1) + node;
                    if node <= m && odds_idx < odds_len {
                        val += *exp_m_ptr.add(ebase + lane) * *odds_ptr.add(odds_idx);
                    }
                }
            }
            *null2_ptr.add(x) = val + insert_total + xfactor;
        }
    }
}

/// Old p_null2_correction without per-row normalization (incorrect).
#[allow(dead_code)]
fn p_null2_correction(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize, // alphabet K
    dsq: &[u8],
    ienv: usize,
    jenv: usize,
    rsc: &[Vec<f32>], // profile emission scores: rsc[x][k*2] = msc(k,x) log-odds
) -> f32 {
    let ld = jenv - ienv + 1;
    let bck_n0 = bck.xmx(0, PXN);
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let bck_scale0 = bck.scale[0];

    // Step 1: Accumulate expected state usage from full-matrix posteriors
    let mut exp_m = vec![0.0_f32; m + 1];
    let mut exp_i = vec![0.0_f32; m + 1];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;

    for i in ienv..=jenv {
        let dp_scale = ((fwd.scale[i] + bck.scale[i] - bck_scale0) as f32).exp() * inv_total;

        for node in 1..=m {
            let mp = fwd.mmx(i, node) * bck.mmx(i, node) * dp_scale;
            let ip = fwd.imx(i, node) * bck.imx(i, node) * dp_scale;
            exp_m[node] += mp;
            exp_i[node] += ip;
        }

        let sp_scale = ((fwd.scale[i - 1] + bck.scale[i] - bck_scale0) as f32).exp() * inv_total;
        exp_n += fwd.xmx(i - 1, PXN) * bck.xmx(i, PXN) * sp_scale;
        exp_j += fwd.xmx(i - 1, PXJ) * bck.xmx(i, PXJ) * sp_scale;
        exp_c += fwd.xmx(i - 1, PXC) * bck.xmx(i, PXC) * sp_scale;
    }

    // Step 2: Normalize to frequencies
    let norm = 1.0 / ld as f32;
    for node in 1..=m {
        exp_m[node] *= norm;
        exp_i[node] *= norm;
    }
    exp_n *= norm;
    exp_c *= norm;
    exp_j *= norm;
    let xfactor = exp_n + exp_c + exp_j;

    // Step 3: Insert total
    let mut insert_total = 0.0_f32;
    for node in 1..m {
        insert_total += exp_i[node];
    }

    // Step 4: Compute null2 odds ratios and correction
    let mut correction = 0.0_f32;
    for i in ienv..=jenv {
        let x = dsq[i] as usize;
        if x >= k {
            continue;
        }

        let mut val = 0.0_f32;
        for node in 1..=m {
            if x < rsc.len() && node * 2 < rsc[x].len() {
                let msc = rsc[x][node * 2]; // log-odds emission score
                if msc > f32::NEG_INFINITY {
                    val += exp_m[node] * msc.exp();
                }
            }
        }
        val += insert_total + xfactor;
        if val > 0.0 {
            correction += val.ln();
        }
    }
    correction
}
