//! Output formatting utilities to match C printf behavior.

/// Format a float like C's `%9.2g` — 9 characters wide, 2 significant digits.
/// C's %g trims trailing zeros and uses exponential when exp < -4 or >= precision.
pub fn fmt_evalue(val: f64) -> String {
    if val == 0.0 {
        return format!("{:>9}", "0");
    }
    if val.is_infinite() {
        return format!("{:>9}", "inf");
    }
    if val.is_nan() {
        return format!("{:>9}", "nan");
    }

    let abs_val = val.abs();
    let exp = abs_val.log10().floor() as i32;

    // C's %g with precision 2: use exponential if exp < -4 or exp >= 2
    if exp >= -4 && exp < 2 {
        // Fixed notation
        // Number of decimal places = precision - 1 - exp (but min 0)
        let decimals = (1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, val);
        // Trim trailing zeros after decimal point (like C's %g)
        let s = if s.contains('.') {
            let s = s.trim_end_matches('0');
            let s = s.trim_end_matches('.');
            s.to_string()
        } else {
            s
        };
        format!("{:>9}", s)
    } else {
        // Exponential notation
        // Format with 1 decimal place, then trim trailing zeros
        let mantissa = val / 10.0_f64.powi(exp);
        let exp_str = format!("e-{:02}", -exp);
        if exp > 0 {
            let exp_str = format!("e+{:02}", exp);
            let m = format!("{:.1}", mantissa);
            let m = m.trim_end_matches('0').trim_end_matches('.');
            format!("{:>9}", format!("{}{}", m, exp_str))
        } else {
            let m = format!("{:.1}", mantissa);
            let m = m.trim_end_matches('0').trim_end_matches('.');
            format!("{:>9}", format!("{}{}", m, exp_str))
        }
    }
}

/// Format a score like C's `%6.1f`.
pub fn fmt_score(val: f32) -> String {
    format!("{:6.1}", val)
}

/// Format a bias like C's `%5.1f`.
pub fn fmt_bias(val: f32) -> String {
    format!("{:5.1}", val)
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
        assert_eq!(fmt_evalue(10.0), "       10");  // Fixed, no trailing .0
        assert_eq!(fmt_evalue(0.5), "      0.5");
    }
}

use std::io::Write;
use crate::tophits::{TopHits, P7_IS_REPORTED};

/// Write per-sequence tabular output (--tblout format).
pub fn write_tblout<W: Write>(f: &mut W, qname: &str, qacc: Option<&str>, th: &TopHits, z: f64) {
    writeln!(f, "#                                                               --- full sequence ---- --- best 1 domain ---- --- domain number estimation ----").unwrap();
    writeln!(f, "# target name        accession  query name           accession    E-value  score  bias   E-value  score  bias   exp reg clu  ov env dom rep inc description of target").unwrap();
    writeln!(f, "#------------------- ---------- -------------------- ---------- --------- ------ ----- --------- ------ -----   --- --- --- --- --- --- --- --- ---------------------").unwrap();

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 { continue; }
        let evalue = z * hit.lnp.exp();
        let dom_evalue = if !hit.dcl.is_empty() { z * hit.dcl[0].lnp.exp() } else { evalue };
        let dom_score = if !hit.dcl.is_empty() { hit.dcl[0].bitscore } else { hit.score };
        writeln!(f,
            "{:<20}{:<11}{:<21}{:<11}{:9.2e} {:6.1} {:5.1} {:9.2e} {:6.1} {:5.1} {:5.1} {:3} {:3} {:3} {:3} {:3} {:3} {:3} {}",
            hit.name, if hit.acc.is_empty() { "-" } else { &hit.acc },
            qname, qacc.unwrap_or("-"),
            evalue, hit.score, hit.bias, dom_evalue, dom_score, hit.bias,
            hit.nexpected, hit.ndom, 0, 0, hit.ndom, hit.ndom, hit.nreported, hit.nincluded,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        ).unwrap();
    }
}

/// Write per-domain tabular output (--domtblout format).
pub fn write_domtblout<W: Write>(f: &mut W, qname: &str, qacc: Option<&str>, th: &TopHits, z: f64, domz: f64) {
    writeln!(f, "#                                                                            --- full sequence --- -------------- this domain -------------   hmm coord   ali coord   env coord").unwrap();
    writeln!(f, "# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target").unwrap();
    writeln!(f, "#------------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------").unwrap();

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 { continue; }
        let evalue = z * hit.lnp.exp();
        for (di, dom) in hit.dcl.iter().enumerate() {
            let dom_evalue = domz * dom.lnp.exp();
            writeln!(f,
                "{:<20}{:<11}{:>5} {:<21}{:<11}{:>5} {:9.2e} {:6.1} {:5.1} {:3} {:3} {:9.2e} {:9.2e} {:6.1} {:5.1} {:5} {:5} {:5} {:5} {:5} {:5} {:.2} {}",
                hit.name, if hit.acc.is_empty() { "-" } else { &hit.acc }, 0,
                qname, qacc.unwrap_or("-"), 0,
                evalue, hit.score, hit.bias, di + 1, hit.ndom,
                dom_evalue / z.max(1.0), dom_evalue, dom.bitscore, dom.dombias,
                1, 0, dom.iali, dom.jali, dom.ienv, dom.jenv, 0.95_f32,
                if hit.desc.is_empty() { "-" } else { &hit.desc },
            ).unwrap();
        }
    }
}
