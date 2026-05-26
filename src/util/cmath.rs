//! C math shims used where bit parity with HMMER's C code matters.

/// Easel's `eslCONST_LOG2`.
pub const ESL_CONST_LOG2: f64 = 0.69314718055994529_f64;

/// Easel's `eslCONST_LOG2R`.
pub const ESL_CONST_LOG2R: f64 = 1.44269504088896341_f64;

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[link(name = "m")]
unsafe extern "C" {
    #[link_name = "log"]
    fn c_log(x: f64) -> f64;

    #[link_name = "logf"]
    fn c_logf(x: f32) -> f32;

    #[link_name = "exp"]
    fn c_exp(x: f64) -> f64;

    #[link_name = "expf"]
    fn c_expf(x: f32) -> f32;

    #[link_name = "pow"]
    fn c_pow(x: f64, y: f64) -> f64;

    #[link_name = "sqrt"]
    fn c_sqrt(x: f64) -> f64;
}

/// `log(double)` as used by the C reference on Unix targets.
#[inline]
pub fn c_log_f64(x: f64) -> f64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        return c_log(x);
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.ln()
    }
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

/// `logf(float)` as used by the C reference on Unix targets.
#[inline]
pub fn c_logf_to_f32(x: f32) -> f32 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        return c_logf(x);
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.ln()
    }
}

/// `exp(double)` as used by the C reference on Unix targets.
#[inline]
pub fn c_exp_f64(x: f64) -> f64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        return c_exp(x);
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.exp()
    }
}

/// `exp(double)`, then cast to `f32`, matching C code that calls `exp()`.
#[inline]
pub fn c_exp_to_f32(x: f64) -> f32 {
    c_exp_f64(x) as f32
}

/// `expf(float)` as used by the C reference on Unix targets.
#[inline]
pub fn c_expf_to_f32(x: f32) -> f32 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        return c_expf(x);
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.exp()
    }
}

/// `pow(double, double)` as used by the C reference on Unix targets.
#[inline]
pub fn c_pow_f64(x: f64, y: f64) -> f64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        return c_pow(x, y);
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.powf(y)
    }
}

/// `sqrt(double)` as used by the C reference on Unix targets.
#[inline]
pub fn c_sqrt_f64(x: f64) -> f64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        return c_sqrt(x);
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.sqrt()
    }
}
