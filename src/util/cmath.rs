//! C math shims used where bit parity with HMMER's C code matters.

/// Easel's `eslCONST_LOG2`.
pub const ESL_CONST_LOG2: f64 = std::f64::consts::LN_2;

/// Easel's `eslCONST_LOG2R`.
pub const ESL_CONST_LOG2R: f64 = std::f64::consts::LOG2_E;

/// Natural log wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_log_f64(x: f64) -> f64 {
    x.ln()
}

/// `log(double)`, then cast to `f32`, matching C code that calls `log()`.
#[inline]
pub fn c_log_to_f32(x: f64) -> f32 {
    c_log_f64(x) as f32
}

/// Promote `f32` to `f64`, call C-style `log(double)`, then truncate to `f32`.
#[inline]
pub fn c_log_f32_to_f32(x: f32) -> f32 {
    c_log_to_f32(x as f64)
}

/// `f32` natural log wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_logf_to_f32(x: f32) -> f32 {
    x.ln()
}

/// Exponential wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_exp_f64(x: f64) -> f64 {
    x.exp()
}

/// `exp(double)`, then cast to `f32`, matching C code that calls `exp()`.
#[inline]
pub fn c_exp_to_f32(x: f64) -> f32 {
    c_exp_f64(x) as f32
}

/// `f32` exponential wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_expf_to_f32(x: f32) -> f32 {
    x.exp()
}

/// Power wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_pow_f64(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Square-root wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrappers_use_rust_float_operations() {
        let x = 2.5;
        assert_eq!(c_log_f64(x), x.ln());
        assert_eq!(c_log_to_f32(x), x.ln() as f32);
        assert_eq!(c_log_f32_to_f32(x as f32), (x as f32 as f64).ln() as f32);
        assert_eq!(c_logf_to_f32(x as f32), (x as f32).ln());
        assert_eq!(c_exp_f64(x), x.exp());
        assert_eq!(c_exp_to_f32(x), x.exp() as f32);
        assert_eq!(c_expf_to_f32(x as f32), (x as f32).exp());
        assert_eq!(c_pow_f64(x, 3.0), x.powf(3.0));
        assert_eq!(c_sqrt_f64(x), x.sqrt());
    }
}
