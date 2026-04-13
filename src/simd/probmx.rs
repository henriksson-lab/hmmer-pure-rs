//! Probability-space DP matrix for SIMD Forward/Backward results.
//! Stores per-position special states (E/N/J/B/C) and cumulative log-scale.
//! Used for domain decoding without needing full per-M-state DP rows.

/// Special state indices in xmx array.
pub const PXE: usize = 0;
pub const PXN: usize = 1;
pub const PXJ: usize = 2;
pub const PXB: usize = 3;
pub const PXC: usize = 4;
const NXCELLS: usize = 5;

/// Probability-space DP matrix (parser mode: specials + scale only).
pub struct ProbMx {
    pub l: usize,
    /// Special states: xmx[(l+1) * 5], indexed as xmx[i * 5 + state]
    pub xmx: Vec<f32>,
    /// Cumulative log-scale per position (f64 for precision)
    pub scale: Vec<f64>,
}

impl ProbMx {
    pub fn new(l: usize) -> Self {
        ProbMx {
            l,
            xmx: vec![0.0; (l + 1) * NXCELLS],
            scale: vec![0.0; l + 1],
        }
    }

    #[inline]
    pub fn xmx(&self, i: usize, s: usize) -> f32 {
        self.xmx[i * NXCELLS + s]
    }

    #[inline]
    pub fn set_xmx(&mut self, i: usize, s: usize, val: f32) {
        self.xmx[i * NXCELLS + s] = val;
    }
}

/// Domain decoding from probability-space Forward/Backward matrices.
/// Port of p_domain_decoding() — computes btot, etot, mocc using the
/// per-position specials and cumulative scales from parser-mode SIMD.
///
/// Matches C's p7_DomainDecoding() normalization:
///   inv_total = 1 / bck_N[0]
///   b_scale = exp(fwd.scale[i-1] + bck.scale[i-1] - bck.scale[0]) * inv_total
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
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let bck_scale0 = bck.scale[0];

    let mut btot = vec![0.0_f32; l + 1];
    let mut etot = vec![0.0_f32; l + 1];
    let mut mocc = vec![0.0_f32; l + 1];

    for i in 1..=l {
        // B: fwd[i-1][B] * bck[i-1][B], with scale correction
        let b_scale = ((fwd.scale[i - 1] + bck.scale[i - 1] - bck_scale0) as f32).exp() * inv_total;
        let b = fwd.xmx(i - 1, PXB) * bck.xmx(i - 1, PXB) * b_scale;
        btot[i] = btot[i - 1] + b;

        // E: fwd[i][E] * bck[i][E], with scale correction
        let e_scale = ((fwd.scale[i] + bck.scale[i] - bck_scale0) as f32).exp() * inv_total;
        let e = fwd.xmx(i, PXE) * bck.xmx(i, PXE) * e_scale;
        etot[i] = etot[i - 1] + e;

        // mocc = 1 - pN - pJ - pC (loop posteriors use fwd[i-1] * bck[i] * transition)
        let sp_scale = ((fwd.scale[i - 1] + bck.scale[i] - bck_scale0) as f32).exp() * inv_total;
        let pn = fwd.xmx(i - 1, PXN) * bck.xmx(i, PXN) * n_loop * sp_scale;
        let pj = fwd.xmx(i - 1, PXJ) * bck.xmx(i, PXJ) * j_loop * sp_scale;
        let pc = fwd.xmx(i - 1, PXC) * bck.xmx(i, PXC) * c_loop * sp_scale;
        mocc[i] = (1.0 - pn - pj - pc).max(0.0);
    }

    (btot, etot, mocc)
}
