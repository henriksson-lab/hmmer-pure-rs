//! C math shims used where bit parity with HMMER's C code matters.

/// Easel's `eslCONST_LOG2`.
pub const ESL_CONST_LOG2: f64 = std::f64::consts::LN_2;

/// Easel's `eslCONST_LOG2R`.
pub const ESL_CONST_LOG2R: f64 = std::f64::consts::LOG2_E;

#[cfg(all(unix, not(target_arch = "wasm32")))]
#[link(name = "m")]
unsafe extern "C" {
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

/// `logf(float)` as used by the C reference on Unix targets.
#[inline]
pub fn c_logf_to_f32(x: f32) -> f32 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        c_logf(x)
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
        c_exp(x)
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
        c_expf(x)
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
        c_pow(x, y)
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
        c_sqrt(x)
    }

    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        x.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::c_log_f64;

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[link(name = "m")]
    unsafe extern "C" {
        #[link_name = "log"]
        fn oracle_c_log(x: f64) -> f64;
    }

    #[test]
    fn rust_log_matches_c_log_on_this_platform() {
        #[cfg(all(unix, not(target_arch = "wasm32")))]
        {
            let special_values = [
                0.0,
                -0.0,
                1.0,
                2.0,
                10.0,
                f64::MIN_POSITIVE,
                f64::from_bits(1),
                f64::from_bits(0x000f_ffff_ffff_ffff),
                f64::from_bits(0x0010_0000_0000_0000),
                f64::from_bits(0x3fe0_0000_0000_0000),
                f64::from_bits(0x3fef_ffff_ffff_ffff),
                f64::from_bits(0x3ff0_0000_0000_0001),
                f64::from_bits(0x3ff8_0000_0000_0000),
                f64::from_bits(0x4000_0000_0000_0001),
                f64::from_bits(0x7fef_ffff_ffff_ffff),
                f64::INFINITY,
            ];

            for x in special_values {
                assert_same_log_value(x);
            }

            let mantissas = [
                0x0000_0000_0000_0,
                0x0000_0000_0000_1,
                0x0000_0000_0000_2,
                0x0000_0000_0000_3,
                0x0000_0000_0000_4,
                0x0000_0000_0000_8,
                0x0000_0000_0001_0,
                0x0000_0000_0100_0,
                0x0000_0100_0000_0,
                0x0001_0000_0000_0,
                0x000f_ffff_ffff_f,
                0x0008_0000_0000_0,
                0x0007_ffff_ffff_f,
            ];

            for exp in 0_u64..=0x7fe {
                for mantissa in mantissas {
                    assert_same_log_bits((exp << 52) | mantissa);
                }
            }
        }
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn assert_same_log_value(x: f64) {
        let rust = c_log_f64(x);
        let c = unsafe { oracle_c_log(x) };
        assert_eq!(
            rust.to_bits(),
            c.to_bits(),
            "log({x:?}) differed: rust={rust:?} c={c:?}"
        );
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn assert_same_log_bits(bits: u64) {
        let x = f64::from_bits(bits);
        if x.is_sign_positive() && !x.is_nan() {
            assert_same_log_value(x);
        }
    }
}
