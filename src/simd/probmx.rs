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

/// Return the in-slice offset (in `f32` elements) needed to reach the next
/// 16-byte aligned boundary inside `v`. Used to align striped DP storage for
/// `_mm_load_ps` / `_mm_store_ps`.
fn aligned_f32_offset(v: &[f32]) -> usize {
    let misalignment = (v.as_ptr() as usize) & 15;
    if misalignment == 0 {
        0
    } else {
        (16 - misalignment) / std::mem::size_of::<f32>()
    }
}

/// Allocate a `Vec<f32>` of at least `len` usable elements plus 4 cells of
/// slack so the returned offset gives a 16-byte aligned interior pointer.
fn aligned_striped_storage(len: usize) -> (Vec<f32>, usize) {
    if len == 0 {
        return (Vec::new(), 0);
    }
    let v = vec![0.0; len + 4];
    let offset = aligned_f32_offset(&v);
    (v, offset)
}

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
    /// Per-row scale factor, matching `P7_OMX xmx[p7X_SCALE]`.
    pub row_scale: Vec<f32>,
    /// Full DP rows (optional): dp[(l+1) * (m+1) * 3]
    /// dp[i * row_width + k * 3 + s] for M(0)/I(1)/D(2) at position i, node k.
    pub dp: Vec<f32>,
    /// Full DP rows in HMMER's striped SIMD layout:
    /// striped_dp[i * striped_row_width + q * 12 + s * 4 + lane].
    pub striped_dp: Vec<f32>,
    pub striped_dp_offset: usize,
    row_width: usize,
    striped_row_width: usize,
    q: usize,
    pub has_dp: bool,
    pub has_own_scales: bool,
}

impl ProbMx {
    /// Allocate a parser-mode `ProbMx` (specials + per-row scale only, no DP rows).
    /// Counterpart of `p7_omx_Create(allocM, 0, L)` in C — used by `*Parser()`
    /// kernels that only keep special-state scores.
    pub fn new(l: usize) -> Self {
        ProbMx {
            m: 0,
            l,
            xmx: vec![0.0; (l + 1) * NXCELLS],
            scale: vec![0.0; l + 1],
            row_scale: vec![1.0; l + 1],
            dp: Vec::new(),
            striped_dp: Vec::new(),
            striped_dp_offset: 0,
            row_width: 0,
            striped_row_width: 0,
            q: 0,
            has_dp: false,
            has_own_scales: false,
        }
    }

    /// Allocate a full-matrix `ProbMx` (specials + scale + striped DP rows).
    /// Counterpart of `p7_omx_Create(M, L, L)` — sized for posterior decoding,
    /// optimal accuracy, or null2 work.
    pub fn new_full(m: usize, l: usize) -> Self {
        let row_width = (m + 1) * DP_CELLS_PER_K;
        let q = m.div_ceil(4);
        let striped_row_width = q * DP_CELLS_PER_K * 4;
        let (striped_dp, striped_dp_offset) = aligned_striped_storage((l + 1) * striped_row_width);
        ProbMx {
            m,
            l,
            xmx: vec![0.0; (l + 1) * NXCELLS],
            scale: vec![0.0; l + 1],
            row_scale: vec![1.0; l + 1],
            dp: Vec::new(),
            striped_dp,
            striped_dp_offset,
            row_width,
            striped_row_width,
            q,
            has_dp: true,
            has_own_scales: false,
        }
    }

    /// Reuse this matrix as parser-only storage (specials + scales, no DP rows).
    /// Analogue of `p7_omx_Reuse` + a parser-shape `p7_omx_GrowTo` in C.
    pub fn resize_parser(&mut self, l: usize) {
        self.m = 0;
        self.l = l;
        self.row_width = 0;
        self.striped_row_width = 0;
        self.q = 0;
        self.has_dp = false;
        self.has_own_scales = false;

        self.xmx.resize((l + 1) * NXCELLS, 0.0);
        self.scale.resize(l + 1, 0.0);
        self.row_scale.resize(l + 1, 1.0);
        self.dp.clear();
        self.striped_dp.clear();
        self.striped_dp_offset = 0;
    }

    /// Reuse this matrix as a full DP matrix sized for `(m, l)`.
    /// Analogue of `p7_omx_Reuse` + `p7_omx_GrowTo(M, L, L)`. Hot Forward/
    /// Backward paths overwrite every active DP row; the same-size reuse
    /// fast-path avoids a full memset but always zeros the row-0 invariant.
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
        let needed = (l + 1) * striped_row_width + 4;
        // Hot Forward/Backward paths overwrite every active DP row. Avoid the
        // full matrix memset on same-size reuse, but preserve Forward's
        // all-zero row-0 invariant below. Newly grown capacity is still
        // initialized so the Vec never contains invalid values.
        self.striped_dp.resize(needed, 0.0);
        self.striped_dp_offset = aligned_f32_offset(&self.striped_dp);
        if striped_row_width > 0 {
            let row0_start = self.striped_dp_offset;
            let row0_end = row0_start + striped_row_width;
            self.striped_dp[row0_start..row0_end].fill(0.0);
        }
    }

    /// Read special-state value `s` at row `i` (E/N/J/B/C).
    #[inline]
    pub fn xmx(&self, i: usize, s: usize) -> f32 {
        self.xmx[i * NXCELLS + s]
    }

    /// Write special-state value `s` at row `i`.
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
            self.striped_dp[self.striped_dp_offset
                + i * self.striped_row_width
                + qi * DP_CELLS_PER_K * 4
                + lane]
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
            self.striped_dp[self.striped_dp_offset
                + i * self.striped_row_width
                + qi * DP_CELLS_PER_K * 4
                + 8
                + lane]
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
            self.striped_dp[self.striped_dp_offset
                + i * self.striped_row_width
                + qi * DP_CELLS_PER_K * 4
                + 4
                + lane]
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
            self.striped_dp[self.striped_dp_offset
                + i * self.striped_row_width
                + qi * DP_CELLS_PER_K * 4
                + lane] = val;
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
            self.striped_dp[self.striped_dp_offset
                + i * self.striped_row_width
                + qi * DP_CELLS_PER_K * 4
                + 8
                + lane] = val;
        } else if !self.dp.is_empty() {
            let idx = i * self.row_width + k * DP_CELLS_PER_K + 1;
            self.dp[idx] = val;
        }
    }

    /// Write a SIMD DP row (`q*3` `__m128` vectors, M/D/I interleaved per
    /// stripe) into row `i` of the striped storage.
    ///
    /// # Safety
    /// `dp_simd` must hold exactly `q * 3` aligned vectors, and row `i` must
    /// be allocated.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn write_simd_row(
        &mut self,
        dp_simd: &[std::arch::x86_64::__m128],
        q: usize,
        _m: usize,
        i: usize,
    ) {
        use std::arch::x86_64::*;
        let row_base = self.striped_dp_offset + i * self.striped_row_width;
        let dst = self.striped_dp.as_mut_ptr().add(row_base);
        let src = dp_simd.as_ptr();
        for qi in 0..q {
            let dst_q = dst.add(qi * DP_CELLS_PER_K * 4);
            let src_q = src.add(qi * DP_CELLS_PER_K);
            _mm_store_ps(dst_q, *src_q);
            _mm_store_ps(dst_q.add(4), *src_q.add(1));
            _mm_store_ps(dst_q.add(8), *src_q.add(2));
        }
    }

    /// Zero out the entire striped DP row `i`.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn zero_simd_row(&mut self, i: usize) {
        let row_base = self.striped_dp_offset + i * self.striped_row_width;
        let row_end = row_base + self.striped_row_width;
        self.striped_dp[row_base..row_end].fill(0.0);
    }

    /// Aligned `*mut f32` to the start of striped DP row `i`.
    ///
    /// # Safety
    /// Row `i` must be allocated.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub unsafe fn striped_row_ptr(&mut self, i: usize) -> *mut f32 {
        self.striped_dp
            .as_mut_ptr()
            .add(self.striped_dp_offset + i * self.striped_row_width)
    }

    /// Number of `f32` cells per striped DP row (== `q * 3 * 4`).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn striped_row_width(&self) -> usize {
        self.striped_row_width
    }

    /// Number of striped stripes `q = ceil(M/4)`.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn q_count(&self) -> usize {
        self.q
    }

    /// Sum the lanes of `state` at row `i` over only the `k <= M` cells.
    /// Tracehash-only helper used to diff striped rows against the C output.
    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_sum(&self, i: usize, state: usize) -> f32 {
        let mut sum = 0.0_f32;
        let row_base = self.striped_dp_offset + i * self.striped_row_width;
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

    /// Sum every lane (including padding past `M`) of `state` at row `i`.
    /// Tracehash-only.
    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_sum_all_lanes(&self, i: usize, state: usize) -> f32 {
        let mut sum = 0.0_f32;
        let row_base = self.striped_dp_offset + i * self.striped_row_width;
        for qi in 0..self.q {
            let off = row_base + qi * DP_CELLS_PER_K * 4 + state * 4;
            for lane in 0..4 {
                sum += self.striped_dp[off + lane];
            }
        }
        sum
    }

    /// Return the 4-lane stripe `(qi, state)` at row `i` as a plain array.
    /// Tracehash-only.
    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_vector(&self, i: usize, state: usize, qi: usize) -> [f32; 4] {
        let off = self.striped_dp_offset
            + i * self.striped_row_width
            + qi * DP_CELLS_PER_K * 4
            + state * 4;
        [
            self.striped_dp[off],
            self.striped_dp[off + 1],
            self.striped_dp[off + 2],
            self.striped_dp[off + 3],
        ]
    }

    /// Sum `state` lanes over a stripe range `[q_start, q_end)` at row `i`,
    /// keeping only `k <= M` cells. Tracehash-only.
    #[cfg(all(feature = "tracehash", target_arch = "x86_64"))]
    pub fn striped_row_state_q_range_sum(
        &self,
        i: usize,
        state: usize,
        q_start: usize,
        q_end: usize,
    ) -> f32 {
        let mut sum = 0.0_f32;
        let row_base = self.striped_dp_offset + i * self.striped_row_width;
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

/// Posterior decoding of domain location from parser-mode Fwd/Bck.
///
/// Ports C `p7_DomainDecoding()` (SSE decoding.c): given parser-mode SIMD
/// Forward `fwd` and Backward `bck`, returns the running `btot`, `etot`, and
/// per-position match-occupancy `mocc[i] = 1 - njcp` arrays. Uses per-row
/// scale factors from `fwd.row_scale` and the `[n_loop, j_loop, c_loop]`
/// transition floats supplied in `njc_loop`.
pub fn p_domain_decoding(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    l: usize,
    njc_loop: [f32; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut btot = Vec::new();
    let mut etot = Vec::new();
    let mut mocc = Vec::new();
    p_domain_decoding_reuse(fwd, bck, m, l, njc_loop, &mut btot, &mut etot, &mut mocc);
    (btot, etot, mocc)
}

/// In-place variant of `p_domain_decoding` that reuses caller-allocated
/// `btot`, `etot`, `mocc` buffers — same algorithm and outputs as
/// `p7_DomainDecoding()`.
pub fn p_domain_decoding_reuse(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    l: usize,
    njc_loop: [f32; 3],
    btot: &mut Vec<f32>,
    etot: &mut Vec<f32>,
    mocc: &mut Vec<f32>,
) {
    #[cfg(not(feature = "tracehash"))]
    let _ = m;
    let n_loop = njc_loop[0];
    let j_loop = njc_loop[1];
    let c_loop = njc_loop[2];

    let bck_n0 = bck.xmx(0, PXN);
    let mut scaleproduct = if bck_n0 > 0.0 {
        (1.0_f64 / bck_n0 as f64) as f32
    } else {
        0.0
    };

    btot.resize(l + 1, 0.0);
    etot.resize(l + 1, 0.0);
    mocc.resize(l + 1, 0.0);
    btot[0] = 0.0;
    etot[0] = 0.0;
    mocc[0] = 0.0;

    for i in 1..=l {
        #[cfg(feature = "tracehash")]
        let scale_before = scaleproduct;
        btot[i] = btot[i - 1];
        let mut b = fwd.xmx(i - 1, PXB);
        b *= bck.xmx(i - 1, PXB);
        b *= fwd.row_scale[i - 1];
        b *= scaleproduct;
        btot[i] += b;

        if bck.has_own_scales {
            scaleproduct *= fwd.row_scale[i - 1] / bck.row_scale[i - 1];
        }
        #[cfg(feature = "tracehash")]
        let scale_after = scaleproduct;

        etot[i] = etot[i - 1];
        let mut e = fwd.xmx(i, PXE);
        e *= bck.xmx(i, PXE);
        e *= fwd.row_scale[i];
        e *= scaleproduct;
        etot[i] += e;

        let mut njcp = fwd.xmx(i - 1, PXN);
        njcp *= bck.xmx(i, PXN);
        njcp *= n_loop;
        njcp *= scaleproduct;

        let mut term = fwd.xmx(i - 1, PXJ);
        term *= bck.xmx(i, PXJ);
        term *= j_loop;
        term *= scaleproduct;
        njcp += term;

        term = fwd.xmx(i - 1, PXC);
        term *= bck.xmx(i, PXC);
        term *= c_loop;
        term *= scaleproduct;
        njcp += term;
        mocc[i] = (1.0_f64 - njcp as f64) as f32;

        #[cfg(feature = "tracehash")]
        if i <= 8 || i == l {
            let mut th = tracehash::th_call!("domain_decoding_step_bits");
            th.input_usize(l);
            th.input_usize(m);
            th.input_usize(i);
            th.output_f32(fwd.xmx(i - 1, PXB));
            th.output_f32(bck.xmx(i - 1, PXB));
            th.output_f32(fwd.row_scale[i - 1]);
            th.output_f32(scale_before);
            th.output_f32(b);
            th.output_f32(btot[i]);
            th.output_f32(scale_after);
            th.output_f32(fwd.xmx(i, PXE));
            th.output_f32(bck.xmx(i, PXE));
            th.output_f32(fwd.row_scale[i]);
            th.output_f32(e);
            th.output_f32(etot[i]);
            th.output_f32(njcp);
            th.output_f32(mocc[i]);
            th.finish();
        }
    }
}

/// Compute the null2 log-correction for a domain envelope from full-matrix
/// probability-space Forward+Backward.
///
/// Wrapper around `p_null2_odds_from_pmx` that converts to odds, exponentiates
/// the residue log-odds, and returns `sum_{i=ienv..=jenv} ln(null2[dsq[i]])`.
/// The per-row normalization handles absolute-magnitude differences between
/// the SIMD Backward and the true values. Conceptually ports
/// `p7_Null2_ByExpectation()` + the trailing correction integration done in
/// `p7_domaindef`.
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

/// Convert a profile's `rsc[x][node*P7P_NR + P7P_MSC]` log-odds emissions to
/// linear odds, packed into a flat `K * (M+1)` array indexed as
/// `odds[x * (M+1) + node]`. Helper for the null2 SSE path.
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

/// Posterior decoding (residue assignment) into a generic `Gmx` from
/// probability-space SIMD Forward + Backward.
///
/// Ports the SSE `p7_Decoding()` (not the generic `p7_GDecoding()`): rows
/// are not renormalized; a Backward matrix that was scaled with Forward row
/// scales reuses `scaleproduct` unchanged. D-state posteriors are zeroed
/// (D is not a residue assignment). Dispatches to a striped fast path when
/// both input matrices have striped DP storage.
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

/// Striped-input fast path for `p_decoding_to_gmx`: scatters striped lanes
/// `(qi, lane) -> k = 1 + qi + lane * q` directly into the dense Gmx storage.
fn p_decoding_to_gmx_striped(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    njc_loop: [f32; 3],
    pp: &mut Gmx,
) {
    let l = fwd.l;
    let fdp = fwd.striped_dp.as_ptr().wrapping_add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().wrapping_add(bck.striped_dp_offset);
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
                scaleproduct *= *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
            }
        }
    }
}

/// Compute null2 odds ratios from full-matrix probability-space Fwd/Bck.
///
/// Ports `p7_Null2_ByExpectation()` (SSE null2.c): accumulates per-state
/// expected usage across the domain envelope `[1..L]`, normalizes by length,
/// and produces null2 odds `null2[x] = (sum_k exp_m[k] * match_odds[x,k]) +
/// insert_total + xfactor` for each canonical residue `x`. Dispatches to a
/// striped-DP fast path when available.
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

/// In-place variant of `p_null2_odds_from_pmx` that reuses caller-allocated
/// `null2`, `exp_m`, `exp_i` scratch buffers. Dispatches to the striped fast
/// path when both inputs have striped DP storage.
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

/// Compute null2 odds directly from a striped posterior matrix `pp`.
///
/// Closer to C `p7_Null2_ByExpectation`: uses the (already-decoded) posterior
/// matrix to accumulate `exp_m`/`exp_i` over rows 1..L (special handling for
/// row 1 to mirror the C memcpy seeding), normalizes by `1/L`, and forms
/// `null2[x] = sum_q (exp_m[qi] * rfv[x][qi] + exp_i[qi]) + xfactor`.
#[cfg(target_arch = "x86_64")]
pub fn p_null2_odds_from_posteriors_reuse(
    pp: &ProbMx,
    k: usize,
    rfv: &[Vec<[f32; 4]>],
    null2: &mut Vec<f32>,
    exp_m: &mut Vec<f32>,
    exp_i: &mut Vec<f32>,
) {
    if pp.striped_dp.is_empty() || pp.l == 0 {
        return;
    }

    let ld = pp.l;
    let q = pp.q;
    let exp_len = q * 4;
    exp_m.resize(exp_len, 0.0);
    exp_i.resize(exp_len, 0.0);

    unsafe {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_load_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
            _mm_shuffle_ps, _mm_store_ss, _mm_storeu_ps,
        };

        let pdp = pp.striped_dp.as_ptr().wrapping_add(pp.striped_dp_offset);
        let px = pp.xmx.as_ptr();
        let exp_m_ptr = exp_m.as_mut_ptr();
        let exp_i_ptr = exp_i.as_mut_ptr();

        let row1 = pdp.add(pp.striped_row_width);
        for qi in 0..q {
            let ebase = qi * 4;
            let pbase = qi * DP_CELLS_PER_K * 4;
            _mm_storeu_ps(exp_m_ptr.add(ebase), _mm_load_ps(row1.add(pbase)));
            _mm_storeu_ps(exp_i_ptr.add(ebase), _mm_load_ps(row1.add(pbase + 8)));
        }
        let mut exp_n = *px.add(NXCELLS + PXN);
        let mut exp_c = *px.add(NXCELLS + PXC);
        let mut exp_j = *px.add(NXCELLS + PXJ);

        for i in 2..=ld {
            let row = pdp.add(i * pp.striped_row_width);
            for qi in 0..q {
                let ebase = qi * 4;
                let pbase = qi * DP_CELLS_PER_K * 4;
                let m = _mm_add_ps(
                    _mm_loadu_ps(exp_m_ptr.add(ebase)),
                    _mm_load_ps(row.add(pbase)),
                );
                let ins = _mm_add_ps(
                    _mm_loadu_ps(exp_i_ptr.add(ebase)),
                    _mm_load_ps(row.add(pbase + 8)),
                );
                _mm_storeu_ps(exp_m_ptr.add(ebase), m);
                _mm_storeu_ps(exp_i_ptr.add(ebase), ins);
            }
            let xrow = i * NXCELLS;
            exp_n += *px.add(xrow + PXN);
            exp_c += *px.add(xrow + PXC);
            exp_j += *px.add(xrow + PXJ);
        }

        let norm = 1.0 / ld as f32;
        let normv = _mm_set1_ps(norm);
        for qi in 0..q {
            let ebase = qi * 4;
            let m = _mm_mul_ps(_mm_loadu_ps(exp_m_ptr.add(ebase)), normv);
            let ins = _mm_mul_ps(_mm_loadu_ps(exp_i_ptr.add(ebase)), normv);
            _mm_storeu_ps(exp_m_ptr.add(ebase), m);
            _mm_storeu_ps(exp_i_ptr.add(ebase), ins);
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;

        null2.resize(k, 1.0);
        for x in 0..k {
            let mut sv = _mm_setzero_ps();
            for qi in 0..q {
                let ebase = qi * 4;
                let odds = _mm_loadu_ps(rfv[x][qi].as_ptr());
                let m = _mm_loadu_ps(exp_m_ptr.add(ebase));
                let ins = _mm_loadu_ps(exp_i_ptr.add(ebase));
                sv = _mm_add_ps(sv, _mm_mul_ps(m, odds));
                sv = _mm_add_ps(sv, ins);
            }
            sv = _mm_add_ps(sv, _mm_shuffle_ps(sv, sv, 0b00_11_10_01));
            sv = _mm_add_ps(sv, _mm_shuffle_ps(sv, sv, 0b01_00_11_10));
            let mut sum = 0.0_f32;
            _mm_store_ss(&mut sum, sv);
            null2[x] = sum + xfactor;
        }
    }
}

/// Fused decode + null2-by-expectation directly from striped Fwd/Bck without
/// materializing a full posterior matrix.
///
/// Equivalent end-state to `p7_Decoding()` followed by `p7_Null2_ByExpectation`,
/// but accumulates `exp_m`/`exp_i` and special-state expectations row-by-row
/// in-place. Has SSE and scalar-fallback bodies. Uses `scaleproduct` rather
/// than `exp(scale)`, matching the SSE convention used elsewhere here.
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

    let fdp = fwd.striped_dp.as_ptr().wrapping_add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().wrapping_add(bck.striped_dp_offset);
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let exp_m_ptr = exp_m.as_mut_ptr();
    let exp_i_ptr = exp_i.as_mut_ptr();

    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let mut scaleproduct = {
        let bck_n0 = bck.xmx(0, PXN);
        if bck_n0 > 0.0 {
            1.0 / bck_n0
        } else {
            0.0
        }
    };

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_load_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
            _mm_shuffle_ps, _mm_store_ss, _mm_storeu_ps,
        };

        for i in 1..=ld {
            let dp_scale = scaleproduct * *fwd.row_scale.as_ptr().add(i);
            let scalev = _mm_set1_ps(dp_scale);
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;

            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                let ebase = qi * 4;

                let fm = _mm_load_ps(fdp.add(fbase));
                let bm = _mm_load_ps(bdp.add(bbase));
                let row_m = _mm_mul_ps(_mm_mul_ps(fm, bm), scalev);
                let old_m = _mm_loadu_ps(exp_m_ptr.add(ebase));
                _mm_storeu_ps(exp_m_ptr.add(ebase), _mm_add_ps(row_m, old_m));

                let fi = _mm_load_ps(fdp.add(fbase + 8));
                let bi = _mm_load_ps(bdp.add(bbase + 8));
                let row_i = _mm_mul_ps(_mm_mul_ps(fi, bi), scalev);
                let old_i = _mm_loadu_ps(exp_i_ptr.add(ebase));
                _mm_storeu_ps(exp_i_ptr.add(ebase), _mm_add_ps(row_i, old_i));
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
                scaleproduct *= *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
            }
        }

        let norm = 1.0 / ld as f32;
        let normv = _mm_set1_ps(norm);
        for qi in 0..q {
            let ebase = qi * 4;
            let m = _mm_mul_ps(_mm_loadu_ps(exp_m_ptr.add(ebase)), normv);
            let ins = _mm_mul_ps(_mm_loadu_ps(exp_i_ptr.add(ebase)), normv);
            _mm_storeu_ps(exp_m_ptr.add(ebase), m);
            _mm_storeu_ps(exp_i_ptr.add(ebase), ins);
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;

        null2.resize(k, 1.0);
        for x in 0..k {
            let mut sv = _mm_setzero_ps();
            for qi in 0..q {
                let ebase = qi * 4;
                let odds = _mm_loadu_ps(rfv[x][qi].as_ptr());
                let m = _mm_loadu_ps(exp_m_ptr.add(ebase));
                let ins = _mm_loadu_ps(exp_i_ptr.add(ebase));
                sv = _mm_add_ps(sv, _mm_mul_ps(m, odds));
                sv = _mm_add_ps(sv, ins);
            }
            sv = _mm_add_ps(sv, _mm_shuffle_ps(sv, sv, 0b00_11_10_01));
            sv = _mm_add_ps(sv, _mm_shuffle_ps(sv, sv, 0b01_00_11_10));
            let mut sum = 0.0_f32;
            _mm_store_ss(&mut sum, sv);
            null2[x] = sum + xfactor;
        }
        return;
    }

    #[cfg(not(target_arch = "x86_64"))]
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
                scaleproduct *= *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
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

/// Compute null2 odds from an already-built generic posterior matrix `pp`.
/// Convenience wrapper around `p_null2_odds_from_gmx_reuse`.
pub fn p_null2_odds_from_gmx(pp: &Gmx, m: usize, k: usize, match_odds: &[f32]) -> Vec<f32> {
    let mut null2 = Vec::new();
    let mut exp_m = Vec::new();
    let mut exp_i = Vec::new();
    p_null2_odds_from_gmx_reuse(pp, m, k, match_odds, &mut null2, &mut exp_m, &mut exp_i);
    null2
}

/// Compute null2 odds from a generic posterior matrix `pp` into caller-owned
/// `null2`/`exp_m`/`exp_i` buffers.
///
/// Accumulates the M and I posteriors with striped indexing (so the result
/// is bit-identical to the striped paths), then forms the same final
/// `null2[x] = sum_k exp_m[k] * match_odds[x, k] + insert_total + xfactor`.
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

/// Striped fast path for `p_null2_odds_from_pmx`: walks the striped Fwd/Bck
/// DP rows directly, performing per-row denominator normalization before
/// accumulating into `exp_m`, `exp_i`, and the special-state expectations.
fn p_null2_odds_from_striped_pmx(
    fwd: &ProbMx,
    bck: &ProbMx,
    m: usize,
    k: usize,
    match_odds: &[f32],
    njc_loop: [f32; 3],
) -> Vec<f32> {
    let ld = fwd.l;
    let fdp = fwd.striped_dp.as_ptr().wrapping_add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().wrapping_add(bck.striped_dp_offset);
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

/// Reuse-buffer variant of `p_null2_odds_from_striped_pmx`. Outputs are
/// identical; only differs in that `null2`, `exp_m`, `exp_i` are passed in.
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
    let fdp = fwd.striped_dp.as_ptr().wrapping_add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().wrapping_add(bck.striped_dp_offset);
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

/// Historical (incorrect) null2 correction that omits the per-row
/// normalization. Retained for reference; not called by current pipeline.
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
