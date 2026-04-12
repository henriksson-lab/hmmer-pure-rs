//! P7_GMX - Generic DP matrix for Forward, Backward, Viterbi.
//! Direct port of p7_gmx.c.

// DP cell state indices
pub const P7G_M: usize = 0; // Match
pub const P7G_I: usize = 1; // Insert
pub const P7G_D: usize = 2; // Delete
pub const P7G_NSCELLS: usize = 3;

// Special state indices
pub const P7G_E: usize = 0;
pub const P7G_N: usize = 1;
pub const P7G_J: usize = 2;
pub const P7G_B: usize = 3;
pub const P7G_C: usize = 4;
pub const P7G_NXCELLS: usize = 5;

/// Generic DP matrix.
#[derive(Debug)]
pub struct Gmx {
    pub m: usize,
    pub l: usize,
    alloc_w: usize, // M+1
    alloc_r: usize, // L+1

    /// DP cells: dp_mem[i * alloc_w * NSCELLS + k * NSCELLS + s]
    pub dp_mem: Vec<f32>,
    /// Row pointers: offsets into dp_mem for each row i
    row_offsets: Vec<usize>,
    /// Special state scores: xmx[i * NXCELLS + s]
    pub xmx: Vec<f32>,
}

impl Gmx {
    /// Create a new DP matrix for model of size M and sequence of length L.
    pub fn new(alloc_m: usize, alloc_l: usize) -> Self {
        let alloc_w = alloc_m + 1;
        let alloc_r = alloc_l + 1;

        let dp_mem = vec![f32::NEG_INFINITY; alloc_r * alloc_w * P7G_NSCELLS];
        let xmx = vec![f32::NEG_INFINITY; alloc_r * P7G_NXCELLS];

        let mut row_offsets = Vec::with_capacity(alloc_r);
        for i in 0..alloc_r {
            row_offsets.push(i * alloc_w * P7G_NSCELLS);
        }

        Gmx {
            m: 0,
            l: 0,
            alloc_w,
            alloc_r,
            dp_mem,
            row_offsets,
            xmx,
        }
    }

    /// Grow matrix to fit model M and sequence L.
    pub fn grow_to(&mut self, m: usize, l: usize) {
        let new_w = m + 1;
        let new_r = l + 1;

        if new_w > self.alloc_w || new_r > self.alloc_r {
            self.alloc_w = new_w.max(self.alloc_w);
            self.alloc_r = new_r.max(self.alloc_r);

            self.dp_mem = vec![f32::NEG_INFINITY; self.alloc_r * self.alloc_w * P7G_NSCELLS];
            self.xmx = vec![f32::NEG_INFINITY; self.alloc_r * P7G_NXCELLS];

            self.row_offsets.resize(self.alloc_r, 0);
            for i in 0..self.alloc_r {
                self.row_offsets[i] = i * self.alloc_w * P7G_NSCELLS;
            }
        }
        self.m = 0;
        self.l = 0;
    }

    /// Access MMX(i,k) - Match state score at position i, node k
    #[inline]
    pub fn mmx(&self, i: usize, k: usize) -> f32 {
        self.dp_mem[self.row_offsets[i] + k * P7G_NSCELLS + P7G_M]
    }

    /// Access IMX(i,k) - Insert state score
    #[inline]
    pub fn imx(&self, i: usize, k: usize) -> f32 {
        self.dp_mem[self.row_offsets[i] + k * P7G_NSCELLS + P7G_I]
    }

    /// Access DMX(i,k) - Delete state score
    #[inline]
    pub fn dmx(&self, i: usize, k: usize) -> f32 {
        self.dp_mem[self.row_offsets[i] + k * P7G_NSCELLS + P7G_D]
    }

    /// Access XMX(i,s) - Special state score
    #[inline]
    pub fn xmx(&self, i: usize, s: usize) -> f32 {
        self.xmx[i * P7G_NXCELLS + s]
    }

    /// Set MMX(i,k)
    #[inline]
    pub fn set_mmx(&mut self, i: usize, k: usize, val: f32) {
        let idx = self.row_offsets[i] + k * P7G_NSCELLS + P7G_M;
        self.dp_mem[idx] = val;
    }

    /// Set IMX(i,k)
    #[inline]
    pub fn set_imx(&mut self, i: usize, k: usize, val: f32) {
        let idx = self.row_offsets[i] + k * P7G_NSCELLS + P7G_I;
        self.dp_mem[idx] = val;
    }

    /// Set DMX(i,k)
    #[inline]
    pub fn set_dmx(&mut self, i: usize, k: usize, val: f32) {
        let idx = self.row_offsets[i] + k * P7G_NSCELLS + P7G_D;
        self.dp_mem[idx] = val;
    }

    /// Set XMX(i,s)
    #[inline]
    pub fn set_xmx(&mut self, i: usize, s: usize, val: f32) {
        self.xmx[i * P7G_NXCELLS + s] = val;
    }

    /// Reuse the matrix for a new calculation.
    pub fn reuse(&mut self) {
        self.m = 0;
        self.l = 0;
    }
}
