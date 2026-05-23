//! Output formatting utilities to match C printf behavior.

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int};

extern "C" {
    fn snprintf(s: *mut c_char, n: usize, format: *const c_char, ...) -> c_int;
}

fn c_snprintf_double(fmt: &[u8], val: f64) -> String {
    let mut buf = [0_i8; 64];
    let n = unsafe {
        snprintf(
            buf.as_mut_ptr(),
            buf.len(),
            fmt.as_ptr().cast::<c_char>(),
            val as c_double,
        )
    };
    if n < 0 {
        return String::new();
    }
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Format an E-value the same way C's `printf("%9.2g", val)` does.
///
/// Width 9, 2 significant digits, exponential form when the decimal exponent
/// is < -4 or >= the precision; trailing zeros and trailing `.` are trimmed.
/// Reproduces C output byte-for-byte for the values HMMER emits.
pub fn fmt_evalue(val: f64) -> String {
    c_snprintf_double(b"%9.2g\0", val)
}

/// Format a bit score using C's `%6.1f` (width 6, 1 decimal).
pub fn fmt_score(val: f32) -> String {
    c_snprintf_double(b"%6.1f\0", val as f64)
}

/// Format a bias-composition correction using C's `%5.1f` (width 5, 1 decimal).
pub fn fmt_bias(val: f32) -> String {
    c_snprintf_double(b"%5.1f\0", val as f64)
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
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_evalue() {
        // Match C %9.2g behavior
        assert_eq!(fmt_evalue(4.2e-24), "  4.2e-24");
        assert_eq!(fmt_evalue(1e-23), "    1e-23");
        assert_eq!(fmt_evalue(7.3e-15), "  7.3e-15");
        assert_eq!(fmt_evalue(0.015), "    0.015");
        assert_eq!(fmt_evalue(2.9e-14), "  2.9e-14");
        assert_eq!(fmt_evalue(10.0), "       10"); // Fixed, no trailing .0
        assert_eq!(fmt_evalue(0.5), "      0.5");
    }
}
