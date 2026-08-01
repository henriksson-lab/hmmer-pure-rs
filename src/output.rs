//! Output formatting utilities to match C printf behavior.

fn fmt_fixed_width(val: f64, width: usize, precision: usize, zero_pad: bool) -> String {
    c_buf_string(if !val.is_finite() {
        pad_width(fmt_nonfinite(val), width)
    } else if zero_pad {
        format!("{val:0width$.precision$}")
    } else {
        format!("{val:width$.precision$}")
    })
}

fn fmt_general(val: f64, width: Option<usize>, precision: usize) -> String {
    let mut s = if !val.is_finite() {
        fmt_nonfinite(val)
    } else if val == 0.0 {
        if val.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        }
    } else {
        let scientific = fmt_scientific_general(val, precision);
        let exp = scientific_exponent(&scientific);
        if exp < -4 || exp >= precision as i32 {
            scientific
        } else {
            let decimals = (precision as i32 - exp - 1).max(0) as usize;
            let fixed = trim_decimal_zeros(format!("{val:.decimals$}"));
            if fixed_integer_digits(&fixed) > precision {
                scientific
            } else {
                fixed
            }
        }
    };
    if let Some(width) = width {
        s = format!("{s:>width$}");
    }
    c_buf_string(s)
}

fn fmt_scientific_general(val: f64, precision: usize) -> String {
    let decimals = precision.saturating_sub(1);
    let raw = format!("{val:.decimals$e}");
    let (mantissa, exponent) = raw.split_once('e').unwrap_or((&raw, "0"));
    format!(
        "{}e{}",
        trim_decimal_zeros(mantissa.to_string()),
        c_exponent(exponent)
    )
}

fn trim_decimal_zeros(mut s: String) -> String {
    if let Some(dot) = s.find('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.len() == dot + 1 {
            s.pop();
        }
    }
    s
}

fn fixed_integer_digits(s: &str) -> usize {
    s.trim_start_matches('-')
        .split_once('.')
        .map_or_else(|| s.trim_start_matches('-').len(), |(int, _)| int.len())
}

fn scientific_exponent(s: &str) -> i32 {
    s.split_once('e')
        .and_then(|(_, exponent)| exponent.parse::<i32>().ok())
        .unwrap_or(0)
}

fn fmt_nonfinite(val: f64) -> String {
    if val.is_nan() {
        "nan".to_string()
    } else if val.is_sign_negative() {
        "-inf".to_string()
    } else {
        "inf".to_string()
    }
}

fn pad_width(s: String, width: usize) -> String {
    if width == 0 {
        s
    } else {
        format!("{s:>width$}")
    }
}

fn c_buf_string(mut s: String) -> String {
    s.truncate(63);
    s
}

fn c_exponent(exponent: &str) -> String {
    let exp = exponent.parse::<i32>().unwrap_or(0);
    let sign = if exp < 0 { '-' } else { '+' };
    let abs = exp.abs();
    if abs < 10 {
        format!("{sign}0{abs}")
    } else {
        format!("{sign}{abs}")
    }
}

/// Format an E-value the same way C's `printf("%9.2g", val)` does.
///
/// Width 9, 2 significant digits, exponential form when the decimal exponent
/// is < -4 or >= the precision; trailing zeros and trailing `.` are trimmed.
/// Reproduces C output byte-for-byte for the values HMMER emits.
pub fn fmt_evalue(val: f64) -> String {
    fmt_general(val, Some(9), 2)
}

/// Format a bit score using C's `%6.1f` (width 6, 1 decimal).
pub fn fmt_score(val: f32) -> String {
    fmt_fixed_width(val as f64, 6, 1, false)
}

/// Format a bias-composition correction using C's `%5.1f` (width 5, 1 decimal).
pub fn fmt_bias(val: f32) -> String {
    fmt_fixed_width(val as f64, 5, 1, false)
}

/// Format a floating-point value using C's default `%g`.
pub fn fmt_g(val: f64) -> String {
    fmt_general(val, None, 6)
}

/// Format a floating-point value using C's `%.3g`.
pub fn fmt_g3(val: f64) -> String {
    fmt_general(val, None, 3)
}

/// Format a floating-point value using C's `%.6f`.
pub fn fmt_fixed6(val: f64) -> String {
    fmt_fixed_width(val, 0, 6, false)
}

/// Format a floating-point value using C's `%.1f`.
pub fn fmt_fixed1(val: f64) -> String {
    fmt_fixed_width(val, 0, 1, false)
}

/// Format a floating-point value using C's `%.0f`.
pub fn fmt_fixed0(val: f64) -> String {
    fmt_fixed_width(val, 0, 0, false)
}

/// Format a floating-point value using C's `%.2f`.
pub fn fmt_fixed2(val: f64) -> String {
    fmt_fixed_width(val, 0, 2, false)
}

/// Format a floating-point value using C's `%.3f`.
pub fn fmt_fixed3(val: f64) -> String {
    fmt_fixed_width(val, 0, 3, false)
}

/// Format a floating-point value using C's `%.4f`.
pub fn fmt_fixed4(val: f64) -> String {
    fmt_fixed_width(val, 0, 4, false)
}

/// Format a floating-point value using C's `%.5f`.
pub fn fmt_fixed5(val: f64) -> String {
    fmt_fixed_width(val, 0, 5, false)
}

/// Format a probability field using C's `%8.5f` (width 8, 5 decimals).
///
/// Matches `printprob` in `hmmer/src/p7_hmmfile.c` which uses `%*.5f`
/// with `fieldwidth=8`.
pub fn fmt_hmm_prob(val: f64) -> String {
    fmt_fixed_width(val, 8, 5, false)
}

/// Format a floating-point value using C's `%8.2f`.
pub fn fmt_width8_2(val: f64) -> String {
    fmt_fixed_width(val, 8, 2, false)
}

/// Format a floating-point value using C's `%5.1f`.
pub fn fmt_width5_1(val: f64) -> String {
    fmt_fixed_width(val, 5, 1, false)
}

/// Format elapsed seconds using C's `%05.2f`.
pub fn fmt_elapsed_seconds(val: f64) -> String {
    fmt_fixed_width(val, 5, 2, true)
}

/// Format a floating-point value using C's `%4.1f`.
pub fn fmt_width4_1(val: f64) -> String {
    fmt_fixed_width(val, 4, 1, false)
}

/// Format a floating-point value using C's `%6.2f`.
pub fn fmt_width6_2(val: f64) -> String {
    fmt_fixed_width(val, 6, 2, false)
}

/// Format a floating-point value using C's `%6.3f`.
pub fn fmt_width6_3(val: f64) -> String {
    fmt_fixed_width(val, 6, 3, false)
}

/// Format a floating-point value using C's `%4.2f`.
pub fn fmt_width4_2(val: f64) -> String {
    fmt_fixed_width(val, 4, 2, false)
}

/// Format a floating-point value using C's `%11.0f`.
pub fn fmt_width11_0(val: f64) -> String {
    fmt_fixed_width(val, 11, 0, false)
}

/// Format a `SystemTime` as HMMER's ctime-style footer date.
pub fn format_hmmer_date(t: std::time::SystemTime) -> String {
    use std::time::UNIX_EPOCH;

    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let (sec, min, hour, day, month, year) = broken_down_time(secs);
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let dow = (((secs / 86400) + 4) % 7) as usize;
    format!(
        "{} {} {:>2} {:02}:{:02}:{:02} {}",
        days[dow],
        months[(month - 1) as usize],
        day,
        hour,
        min,
        sec,
        year
    )
}

fn broken_down_time(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let sec = (secs % 60) as u32;
    let min = ((secs / 60) % 60) as u32;
    let hour = ((secs / 3600) % 24) as u32;
    let mut days = secs / 86400;
    let mut year = 1970;
    loop {
        let yd = if is_leap(year) { 366 } else { 365 };
        if days < yd {
            break;
        }
        days -= yd;
        year += 1;
    }
    let mdays = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &m in &mdays {
        if days < m {
            break;
        }
        days -= m;
        month += 1;
    }
    (sec, min, hour, (days + 1) as u32, month, year)
}

fn is_leap(y: u32) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_evalue() {
        // Match C %9.2g behavior
        assert_eq!(fmt_evalue(-0.0), "       -0");
        assert_eq!(fmt_evalue(4.2e-24), "  4.2e-24");
        assert_eq!(fmt_evalue(1e-23), "    1e-23");
        assert_eq!(fmt_evalue(7.3e-15), "  7.3e-15");
        assert_eq!(fmt_evalue(9.999e-5), "   0.0001");
        assert_eq!(fmt_evalue(0.015), "    0.015");
        assert_eq!(fmt_evalue(2.9e-14), "  2.9e-14");
        assert_eq!(fmt_evalue(10.0), "       10"); // Fixed, no trailing .0
        assert_eq!(fmt_evalue(99.95), "    1e+02");
        assert_eq!(fmt_evalue(0.5), "      0.5");
        assert_eq!(fmt_evalue(f64::NAN), "      nan");
    }

    #[test]
    fn general_formatting_uses_c_exponent_shape() {
        assert_eq!(fmt_g(-0.0), "-0");
        assert_eq!(fmt_g(1.0e-5), "1e-05");
        assert_eq!(fmt_g(1234567.0), "1.23457e+06");
        assert_eq!(fmt_g3(0.001234), "0.00123");
        assert_eq!(fmt_g3(9.999e-5), "0.0001");
        assert_eq!(fmt_g3(999.95), "1e+03");
        assert_eq!(fmt_g3(1234.0), "1.23e+03");
        assert_eq!(fmt_g(f64::NAN), "nan");
    }

    #[test]
    fn fixed_width_formatting_preserves_field_shapes() {
        assert_eq!(fmt_score(12.34), "  12.3");
        assert_eq!(fmt_score(f32::NAN), "   nan");
        assert_eq!(fmt_bias(0.0), "  0.0");
        assert_eq!(fmt_hmm_prob(0.25), " 0.25000");
        assert_eq!(fmt_hmm_prob(f64::NAN), "     nan");
        assert_eq!(fmt_width11_0(42.0), "         42");
        assert_eq!(fmt_width11_0(f64::NAN), "        nan");
    }

    #[test]
    fn elapsed_seconds_uses_c_zero_padded_rounding() {
        assert_eq!(fmt_elapsed_seconds(0.0), "00.00");
        assert_eq!(fmt_elapsed_seconds(1.2), "01.20");
        assert_eq!(fmt_elapsed_seconds(12.345), "12.35");
        assert_eq!(fmt_elapsed_seconds(123.456), "123.46");
        assert_eq!(fmt_elapsed_seconds(f64::INFINITY), "  inf");
    }

    #[test]
    fn hmmer_date_has_ctime_shape_without_newline() {
        let s = format_hmmer_date(std::time::UNIX_EPOCH);
        assert_eq!(s.len(), "Thu Jan  1 00:00:00 1970".len());
        assert!(!s.contains('\n'));
        assert_eq!(&s[3..4], " ");
        assert_eq!(&s[7..8], " ");
        assert_eq!(&s[10..11], " ");
        assert_eq!(&s[19..20], " ");
    }

    #[cfg(not(unix))]
    #[test]
    fn hmmer_date_non_unix_fallback_is_utc() {
        assert_eq!(
            format_hmmer_date(std::time::UNIX_EPOCH),
            "Thu Jan  1 00:00:00 1970"
        );
    }
}
