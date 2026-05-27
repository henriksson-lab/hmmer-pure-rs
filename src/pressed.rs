//! Read/write C HMMER pressed database format (.h3f, .h3p, .h3i).
//! Enables reading databases created by C hmmpress.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::alphabet::AlphabetType;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::Hmm;
use crate::hmmfile_binary;
#[cfg(target_arch = "x86_64")]
use crate::simd::oprofile::AlignedF32x4;
use crate::simd::oprofile::{nqb, nqf, nqw, OProfile};

// Magic numbers from C HMMER impl_sse/io.c, v3f format.
const V3F_FMAGIC: u32 = 0xb3e6e6f3; // .h3f sentinel
#[allow(dead_code)]
const V3F_PMAGIC: u32 = 0xb3e6f0f3; // .h3p sentinel (used for future .h3p reader)

const SSI_V30_MAGIC: u32 = 0xd3d3c9b3;
const P7_MAXABET: usize = 20;
const P7_NOFFSETS: usize = 3;
const P7O_EXTRA_SB: usize = 17;

/// Read a native-endian `u32` from `r`.
fn read_u32_ne<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_ne_bytes(buf))
}

fn read_record_magic_ne<R: Read>(r: &mut R) -> HmmerResult<Option<u32>> {
    let mut buf = [0u8; 4];
    match r.read(&mut buf[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!(),
        Err(e) => return Err(HmmerError::Io(e)),
    }
    r.read_exact(&mut buf[1..]).map_err(HmmerError::Io)?;
    Ok(Some(u32::from_ne_bytes(buf)))
}

/// Read a native-endian `i32` from `r`.
fn read_i32_ne<R: Read>(r: &mut R) -> HmmerResult<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i32::from_ne_bytes(buf))
}

/// Read a native-endian `f32` from `r`.
fn read_f32_ne<R: Read>(r: &mut R) -> HmmerResult<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(f32::from_ne_bytes(buf))
}

/// Read one byte from `r`.
fn read_u8<R: Read>(r: &mut R) -> HmmerResult<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf[0])
}

/// Read exactly `n` bytes from `r` into a fresh `Vec<u8>`.
fn read_bytes<R: Read>(r: &mut R, n: usize) -> HmmerResult<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf)
}

fn read_nonnegative_len<R: Read>(r: &mut R, field: &str) -> HmmerResult<usize> {
    let len = read_i32_ne(r)?;
    if len < 0 {
        return Err(HmmerError::Format(format!(
            "Negative {field} in pressed profile record: {len}"
        )));
    }
    Ok(len as usize)
}

fn read_len_prefixed_cstr<R: Read>(r: &mut R, len: usize, field: &str) -> HmmerResult<String> {
    if len == 0 {
        return Ok(String::new());
    }
    let bytes = read_bytes(r, len + 1)?;
    if bytes.last() != Some(&0) {
        return Err(HmmerError::Format(format!(
            "{field} is missing trailing NUL terminator"
        )));
    }
    if bytes[..len].contains(&0) {
        return Err(HmmerError::Format(format!(
            "{field} contains embedded NUL byte"
        )));
    }
    Ok(String::from_utf8_lossy(&bytes[..len]).to_string())
}

fn require_nonempty_required_string(value: String, field: &str) -> HmmerResult<String> {
    if value.is_empty() {
        return Err(HmmerError::Format(format!("{field} is empty")));
    }
    Ok(value)
}

fn validate_pressed_record_header(m: i32, abc_type: i32, sidecar: &str) -> HmmerResult<usize> {
    if m <= 0 {
        return Err(HmmerError::Format(format!(
            "Invalid {sidecar} model length: {m}"
        )));
    }
    if !matches!(abc_type, 1..=3) {
        return Err(HmmerError::Format(format!(
            "Invalid {sidecar} alphabet code: {abc_type}"
        )));
    }
    Ok(m as usize)
}

fn alphabet_type_code(abc_type: AlphabetType) -> HmmerResult<i32> {
    match abc_type {
        AlphabetType::Rna => Ok(1),
        AlphabetType::Dna => Ok(2),
        AlphabetType::Amino => Ok(3),
        AlphabetType::Unknown => Err(HmmerError::Format(
            "Cannot press HMM with unknown alphabet".to_string(),
        )),
    }
}

fn write_i32_ne<W: Write>(w: &mut W, v: i32) -> HmmerResult<()> {
    w.write_all(&v.to_ne_bytes()).map_err(HmmerError::Io)
}

fn write_u32_ne<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_ne_bytes()).map_err(HmmerError::Io)
}

fn write_f32_ne<W: Write>(w: &mut W, v: f32) -> HmmerResult<()> {
    w.write_all(&v.to_ne_bytes()).map_err(HmmerError::Io)
}

fn write_len_prefixed_cstr<W: Write>(w: &mut W, s: &str) -> HmmerResult<()> {
    write_i32_ne(w, s.len() as i32)?;
    w.write_all(s.as_bytes()).map_err(HmmerError::Io)?;
    w.write_all(&[0]).map_err(HmmerError::Io)
}

fn write_optional_len_prefixed_cstr<W: Write>(w: &mut W, s: Option<&str>) -> HmmerResult<()> {
    match s {
        Some(s) if !s.is_empty() => write_len_prefixed_cstr(w, s),
        _ => write_i32_ne(w, 0),
    }
}

fn write_annotation<W: Write>(w: &mut W, annotation: Option<&[u8]>, m: usize) -> HmmerResult<()> {
    let len = m + 2;
    match annotation {
        // C dumps `om->rf/mm/cs/consensus` of width M+2 (impl_sse/io.c:146-149).
        // Those arrays are `strcpy`'d from the gm strings (p7_oprofile.c:1112),
        // so index M+1 is the copied `'\0'` terminator. Our in-memory arrays
        // keep a `' '` (space) sentinel at index M+1; force the trailing byte to
        // a NUL to match C byte-for-byte (same class as the hmmfile_binary fix).
        Some(bytes) => {
            let mut buf = vec![0u8; len];
            let n = bytes.len().min(len);
            buf[..n].copy_from_slice(&bytes[..n]);
            buf[len - 1] = 0;
            w.write_all(&buf).map_err(HmmerError::Io)
        }
        None => {
            let zeros = vec![0u8; len];
            w.write_all(&zeros).map_err(HmmerError::Io)
        }
    }
}

/// Write one C HMMER3/f `.h3f` MSV-filter record.
pub fn write_h3f_record<W: Write>(
    w: &mut W,
    hmm: &Hmm,
    om: &OProfile,
    offsets: [i64; P7_NOFFSETS],
) -> HmmerResult<()> {
    let abc_type = alphabet_type_code(hmm.abc_type)?;
    let q16 = nqb(om.m);
    let q16x = q16 + P7O_EXTRA_SB;

    write_u32_ne(w, V3F_FMAGIC)?;
    write_i32_ne(w, om.m as i32)?;
    write_i32_ne(w, abc_type)?;
    write_len_prefixed_cstr(w, &om.name)?;
    // C `impl_sse/io.c:102` writes `om->max_length` here (the per-model upper
    // bound on emitted length, from the HMM `MAXL` line), NOT the configured L.
    write_i32_ne(w, om.max_length)?;
    w.write_all(&[om.tbm_b, om.tec_b, om.tjb_b])
        .map_err(HmmerError::Io)?;
    write_f32_ne(w, om.scale_b)?;
    w.write_all(&[om.base_b, om.bias_b])
        .map_err(HmmerError::Io)?;

    for x in 0..om.abc_kp {
        for q in 0..q16x {
            w.write_all(&om.sbv[x][q]).map_err(HmmerError::Io)?;
        }
    }
    for x in 0..om.abc_kp {
        for q in 0..q16 {
            w.write_all(&om.rbv[x][q]).map_err(HmmerError::Io)?;
        }
    }
    for value in om.evparam {
        write_f32_ne(w, value)?;
    }
    for offset in offsets {
        w.write_all(&offset.to_ne_bytes()).map_err(HmmerError::Io)?;
    }
    // C writes p7_MAXABET(=20) compo floats (impl_sse/io.c:118). For nucleic
    // (K=4) models the unused tail entries are 0.0 in C's array, whereas our
    // `om.compo` keeps COMPO_UNSET (-1.0) there. Emit the real composition for
    // [0..K) and 0.0 for the tail so the .h3f matches C byte-for-byte. Amino
    // (K=20) uses every entry, so this is a no-op for amino models.
    for (i, value) in om.compo.iter().enumerate() {
        let out = if i < om.abc_k { *value } else { 0.0 };
        write_f32_ne(w, out)?;
    }
    write_u32_ne(w, V3F_FMAGIC)
}

/// Write one C HMMER3/f `.h3p` optimized-profile remainder record.
pub fn write_h3p_record<W: Write>(w: &mut W, hmm: &Hmm, om: &OProfile) -> HmmerResult<()> {
    let abc_type = alphabet_type_code(hmm.abc_type)?;
    let q8 = nqw(om.m);
    let q4 = nqf(om.m);

    write_u32_ne(w, V3F_PMAGIC)?;
    write_i32_ne(w, om.m as i32)?;
    write_i32_ne(w, abc_type)?;
    write_len_prefixed_cstr(w, &om.name)?;
    write_optional_len_prefixed_cstr(w, hmm.acc.as_deref())?;
    write_optional_len_prefixed_cstr(w, hmm.desc.as_deref())?;

    write_annotation(w, hmm.rf.as_deref(), om.m)?;
    write_annotation(w, hmm.mm.as_deref(), om.m)?;
    write_annotation(w, hmm.cs.as_deref(), om.m)?;
    write_annotation(w, hmm.consensus.as_deref(), om.m)?;

    for q in 0..8 * q8 {
        for value in om.twv[q] {
            w.write_all(&value.to_ne_bytes()).map_err(HmmerError::Io)?;
        }
    }
    for x in 0..om.abc_kp {
        for q in 0..q8 {
            for value in om.rwv[x][q] {
                w.write_all(&value.to_ne_bytes()).map_err(HmmerError::Io)?;
            }
        }
    }
    for state in 0..4 {
        for trans in 0..2 {
            w.write_all(&om.xw[state][trans].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }
    write_f32_ne(w, om.scale_w)?;
    w.write_all(&om.base_w.to_ne_bytes())
        .map_err(HmmerError::Io)?;
    w.write_all(&om.ddbound_w.to_ne_bytes())
        .map_err(HmmerError::Io)?;
    write_f32_ne(w, om.ncj_roundoff)?;

    for q in 0..8 * q4 {
        for value in om.tfv[q] {
            write_f32_ne(w, value)?;
        }
    }
    for x in 0..om.abc_kp {
        for q in 0..q4 {
            for value in om.rfv[x][q] {
                write_f32_ne(w, value)?;
            }
        }
    }
    for state in 0..4 {
        for trans in 0..2 {
            write_f32_ne(w, om.xf[state][trans])?;
        }
    }
    for value in om.cutoff {
        write_f32_ne(w, value)?;
    }
    write_f32_ne(w, om.nj)?;
    write_i32_ne(w, om.mode)?;
    write_i32_ne(w, om.l)?;
    write_u32_ne(w, V3F_PMAGIC)
}

/// MSV filter data from .h3f file for one profile.
pub struct MsvFilterData {
    pub name: String,
    pub m: usize,
    pub abc_type: i32,
    pub max_length: i32,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,
    /// Raw SSV/MSV byte scores (flattened __m128i arrays)
    pub msv_scores: Vec<u8>,
    pub evparam: [f32; 6],
    pub offsets: [i64; P7_NOFFSETS],
    pub compo: [f32; 20],
}

fn alphabet_dimensions(abc_type: i32, sidecar: &str) -> HmmerResult<(usize, usize)> {
    match abc_type {
        1 | 2 => Ok((4, 18)),
        3 => Ok((20, 29)),
        _ => Err(HmmerError::Format(format!(
            "Invalid {sidecar} alphabet code: {abc_type}"
        ))),
    }
}

/// Read one MSV filter (`.h3f`) record produced by `hmmpress`.
/// Reads the V3F magic, the SSV/MSV byte score table, and footer magic; mirrors
/// `p7_oprofile_ReadMSV` in `hmmer/src/p7_hmmfile.c`. Returns `Ok(None)` at EOF.
pub fn read_h3f_record<R: Read>(r: &mut R) -> HmmerResult<Option<MsvFilterData>> {
    let Some(magic) = read_record_magic_ne(r)? else {
        return Ok(None);
    };
    if magic != V3F_FMAGIC {
        return Err(HmmerError::Format(format!("Bad .h3f magic: {:#x}", magic)));
    }

    let m = read_i32_ne(r)?;
    let abc_type = read_i32_ne(r)?;
    let m = validate_pressed_record_header(m, abc_type, ".h3f")?;
    let name_len = read_nonnegative_len(r, ".h3f name length")?;
    let name = require_nonempty_required_string(
        read_len_prefixed_cstr(r, name_len, ".h3f name")?,
        ".h3f name",
    )?;
    let max_length = read_i32_ne(r)?;

    let tbm_b = read_u8(r)?;
    let tec_b = read_u8(r)?;
    let tjb_b = read_u8(r)?;
    let scale_b = read_f32_ne(r)?;
    let base_b = read_u8(r)?;
    let bias_b = read_u8(r)?;

    // Read MSV score vectors (raw bytes): sbv has Q16 + p7O_EXTRA_SB
    // vectors per alphabet symbol; rbv has Q16.
    let (_k, k) = alphabet_dimensions(abc_type, ".h3f")?;
    let q = ((m.max(1) - 1) / 16 + 1).max(2);
    let msv_bytes = k * (q + P7O_EXTRA_SB + q) * 16; // sbv + rbv
    let msv_scores = read_bytes(r, msv_bytes)?;

    let mut evparam = [0.0f32; 6];
    for i in 0..6 {
        evparam[i] = read_f32_ne(r)?;
    }

    let mut offsets = [0i64; P7_NOFFSETS];
    for offset in &mut offsets {
        let mut buf = [0u8; std::mem::size_of::<i64>()];
        r.read_exact(&mut buf).map_err(HmmerError::Io)?;
        *offset = i64::from_ne_bytes(buf);
    }

    let mut compo = [0.0f32; P7_MAXABET];
    for i in 0..P7_MAXABET {
        compo[i] = read_f32_ne(r)?;
    }

    // Footer magic
    let footer = read_u32_ne(r)?;
    if footer != V3F_FMAGIC {
        return Err(HmmerError::Format("Bad .h3f footer".to_string()));
    }

    Ok(Some(MsvFilterData {
        name,
        m,
        abc_type,
        max_length,
        tbm_b,
        tec_b,
        tjb_b,
        scale_b,
        base_b,
        bias_b,
        msv_scores,
        evparam,
        offsets,
        compo,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::bg::Bg;
    use crate::hmmfile;
    use crate::hmmfile_binary::write_binary_hmm;
    use crate::profile::{self, Profile, P7_LOCAL};
    use crate::ssi;
    use std::io::Cursor;

    fn push_i32(buf: &mut Vec<u8>, v: i32) {
        buf.extend_from_slice(&v.to_ne_bytes());
    }

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_ne_bytes());
    }

    fn push_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_ne_bytes());
    }

    fn minimal_h3f_record(m: i32, abc_type: i32, name: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        push_u32(&mut buf, V3F_FMAGIC);
        push_i32(&mut buf, m);
        push_i32(&mut buf, abc_type);
        push_i32(&mut buf, name.len() as i32);
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        push_i32(&mut buf, 400);
        buf.extend_from_slice(&[1, 2, 3]);
        push_f32(&mut buf, 4.5);
        buf.extend_from_slice(&[5, 6]);

        let k = if abc_type == 3 { 29 } else { 18 };
        let q = (((m as usize).max(1) - 1) / 16 + 1).max(2);
        let msv_bytes = k * (q + P7O_EXTRA_SB + q) * 16;
        buf.extend((0..msv_bytes).map(|i| (i % 251) as u8));

        for i in 0..6 {
            push_f32(&mut buf, 10.0 + i as f32);
        }
        for i in 0..3 {
            buf.extend_from_slice(&(1000_i64 + i).to_ne_bytes());
        }
        for i in 0..P7_MAXABET {
            push_f32(&mut buf, 100.0 + i as f32);
        }
        push_u32(&mut buf, V3F_FMAGIC);
        buf
    }

    fn fn3_hmm() -> Hmm {
        hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/fn3.hmm"
        )))
        .unwrap()
        .remove(0)
    }

    fn made1_hmm() -> Hmm {
        hmmfile::read_hmm_file(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/MADE1.hmm"
        )))
        .unwrap()
        .remove(0)
    }

    fn oprofile_for_press(hmm: &Hmm) -> OProfile {
        let abc = Alphabet::new(hmm.abc_type);
        let mut bg = Bg::new(&abc);
        bg.set_length(400);
        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        OProfile::convert(&gm)
    }

    fn write_valid_pressed_set(hmm_path: &Path, hmm: &Hmm) {
        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let om = oprofile_for_press(hmm);

        let mut m = std::fs::File::create(&h3m).unwrap();
        write_binary_hmm(&mut m, hmm).unwrap();
        let mut f = std::fs::File::create(&h3f).unwrap();
        write_h3f_record(&mut f, hmm, &om, [0, 0, 0]).unwrap();
        let mut p = std::fs::File::create(&h3p).unwrap();
        write_h3p_record(&mut p, hmm, &om).unwrap();
        ssi::write_hmm_ssi_records(&h3m, &h3i, [(hmm.name.clone(), hmm.acc.clone(), 0)], false)
            .unwrap();
    }

    fn read_be_u32_at(buf: &[u8], offset: usize) -> u32 {
        u32::from_be_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn read_be_u64_at(buf: &[u8], offset: usize) -> u64 {
        u64::from_be_bytes(buf[offset..offset + 8].try_into().unwrap())
    }

    fn write_be_u64_at(buf: &mut [u8], offset: usize, value: u64) {
        buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
    }

    fn write_fixed_string_at(buf: &mut [u8], offset: usize, len: usize, value: &str) {
        let dst = &mut buf[offset..offset + len];
        dst.fill(0);
        let bytes = value.as_bytes();
        dst[..bytes.len().min(len)].copy_from_slice(&bytes[..bytes.len().min(len)]);
    }

    fn write_be_i64_at(buf: &mut [u8], offset: usize, value: i64) {
        buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
    }

    fn h3i_primary_data_offset_position(buf: &[u8]) -> usize {
        let plen = read_be_u32_at(buf, 34) as usize;
        let poffset = read_be_u64_at(buf, 62) as usize;
        poffset + plen + std::mem::size_of::<u16>() + std::mem::size_of::<u64>()
    }

    fn write_two_record_pressed_set(hmm_path: &Path, first: &Hmm, second: &Hmm) {
        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let om = oprofile_for_press(first);

        let mut h3m_bytes = Vec::new();
        write_binary_hmm(&mut h3m_bytes, first).unwrap();
        let second_offset = h3m_bytes.len() as u64;
        write_binary_hmm(&mut h3m_bytes, second).unwrap();
        std::fs::write(&h3m, h3m_bytes).unwrap();

        let mut f = std::fs::File::create(&h3f).unwrap();
        write_h3f_record(&mut f, first, &om, [0, 0, 0]).unwrap();
        let mut p = std::fs::File::create(&h3p).unwrap();
        write_h3p_record(&mut p, first, &om).unwrap();
        ssi::write_hmm_ssi_records(
            &h3m,
            &h3i,
            [
                (first.name.clone(), first.acc.clone(), 0),
                (second.name.clone(), second.acc.clone(), second_offset),
            ],
            false,
        )
        .unwrap();
    }

    fn assert_oprofile_exact(actual: &OProfile, expected: &OProfile) {
        assert_eq!(actual.m, expected.m);
        assert_eq!(actual.l, expected.l);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.abc_k, expected.abc_k);
        assert_eq!(actual.abc_kp, expected.abc_kp);
        assert_eq!(actual.rbv, expected.rbv);
        assert_eq!(actual.sbv, expected.sbv);
        assert_eq!(actual.tbm_b, expected.tbm_b);
        assert_eq!(actual.tec_b, expected.tec_b);
        assert_eq!(actual.tjb_b, expected.tjb_b);
        assert_eq!(actual.scale_b.to_bits(), expected.scale_b.to_bits());
        assert_eq!(actual.base_b, expected.base_b);
        assert_eq!(actual.bias_b, expected.bias_b);
        assert_eq!(actual.rwv, expected.rwv);
        assert_eq!(actual.twv, expected.twv);
        assert_eq!(actual.xw, expected.xw);
        assert_eq!(actual.scale_w.to_bits(), expected.scale_w.to_bits());
        assert_eq!(actual.base_w, expected.base_w);
        assert_eq!(actual.ddbound_w, expected.ddbound_w);
        assert_eq!(
            actual.ncj_roundoff.to_bits(),
            expected.ncj_roundoff.to_bits()
        );
        assert_eq!(actual.rfv, expected.rfv);
        assert_eq!(actual.tfv, expected.tfv);
        assert_eq!(actual.xf, expected.xf);
        assert_eq!(actual.nj.to_bits(), expected.nj.to_bits());
        assert_eq!(actual.mode, expected.mode);
        assert_eq!(actual.evparam, expected.evparam);
        assert_eq!(actual.cutoff, expected.cutoff);
        assert_eq!(actual.compo, expected.compo);
    }

    #[cfg(target_arch = "x86_64")]
    fn msv_score(result: crate::simd::msv_filter::MsvResult) -> Option<u32> {
        match result {
            crate::simd::msv_filter::MsvResult::Ok(sc) => Some(sc.to_bits()),
            crate::simd::msv_filter::MsvResult::Overflow => None,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn vit_score(result: crate::simd::vit_filter::VitResult) -> Option<u32> {
        match result {
            crate::simd::vit_filter::VitResult::Ok(sc) => Some(sc.to_bits()),
            crate::simd::vit_filter::VitResult::Overflow => None,
        }
    }

    #[test]
    fn reads_h3f_msv_scores_with_extra_sbv_vectors() {
        let mut cursor = Cursor::new(minimal_h3f_record(20, 3, "prot"));
        let rec = read_h3f_record(&mut cursor).unwrap().unwrap();

        let q = 2;
        assert_eq!(rec.msv_scores.len(), 29 * (q + P7O_EXTRA_SB + q) * 16);
        assert_eq!(rec.name, "prot");
        assert_eq!(rec.compo[19], 119.0);
        assert!(read_h3f_record(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn reads_all_twenty_composition_floats_for_dna_profiles() {
        let mut cursor = Cursor::new(minimal_h3f_record(7, 2, "dna"));
        let rec = read_h3f_record(&mut cursor).unwrap().unwrap();

        assert_eq!(rec.msv_scores.len(), 18 * (2 + P7O_EXTRA_SB + 2) * 16);
        assert_eq!(rec.compo[0], 100.0);
        assert_eq!(rec.compo[19], 119.0);
    }

    #[test]
    fn read_h3f_rejects_truncated_initial_magic() {
        let mut cursor = Cursor::new(vec![0xb3, 0xe6]);

        let err = match read_h3f_record(&mut cursor) {
            Ok(_) => panic!("truncated .h3f initial magic was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("failed to fill whole buffer"));
    }

    #[test]
    fn read_h3f_rejects_invalid_header_values() {
        let mut cursor = Cursor::new(minimal_h3f_record(0, 3, "bad"));

        let err = match read_h3f_record(&mut cursor) {
            Ok(_) => panic!("invalid .h3f header was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Invalid .h3f model length"));
    }

    #[test]
    fn read_h3f_rejects_embedded_nul_in_name() {
        let mut record = minimal_h3f_record(20, 3, "prot");
        record[16 + 2] = 0;

        let err = match read_h3f_record(&mut Cursor::new(record)) {
            Ok(_) => panic!("embedded NUL in .h3f name was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("embedded NUL"));
    }

    #[test]
    fn read_h3f_rejects_empty_name() {
        let err = match read_h3f_record(&mut Cursor::new(minimal_h3f_record(20, 3, ""))) {
            Ok(_) => panic!("empty .h3f name was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains(".h3f name is empty"));
    }

    #[test]
    fn read_h3p_rejects_missing_name_terminator() {
        let hmm = fn3_hmm();
        let om = oprofile_for_press(&hmm);
        let mut record = Vec::new();
        write_h3p_record(&mut record, &hmm, &om).unwrap();
        let name_len = i32::from_ne_bytes(record[12..16].try_into().unwrap()) as usize;
        record[16 + name_len] = b'X';

        let err = match read_h3p_record(&mut Cursor::new(record)) {
            Ok(_) => panic!("missing .h3p name terminator was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("missing trailing NUL"));
    }

    #[test]
    fn read_h3p_rejects_empty_name() {
        let mut hmm = fn3_hmm();
        hmm.name.clear();
        let om = oprofile_for_press(&hmm);
        let mut record = Vec::new();
        write_h3p_record(&mut record, &hmm, &om).unwrap();

        let err = match read_h3p_record(&mut Cursor::new(record)) {
            Ok(_) => panic!("empty .h3p name was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains(".h3p name is empty"));
    }

    #[test]
    fn pressed_db_available_rejects_partial_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let hmm = dir.path().join("models.hmm");
        std::fs::write(&hmm, b"HMMER3/f\nNAME  x\nLENG  1\n//\n").unwrap();
        std::fs::write(format!("{}.h3m", hmm.display()), b"partial").unwrap();

        let err = pressed_db_available(&hmm).unwrap_err();
        assert!(err.to_string().contains("Incomplete pressed database"));
    }

    #[test]
    fn pressed_db_available_rejects_bad_h3f_magic() {
        let dir = tempfile::tempdir().unwrap();
        let hmm = dir.path().join("models.hmm");
        std::fs::write(&hmm, b"HMMER3/f\nNAME  x\nLENG  1\n//\n").unwrap();
        std::fs::write(format!("{}.h3f", hmm.display()), 0_u32.to_ne_bytes()).unwrap();
        std::fs::write(format!("{}.h3p", hmm.display()), 0_u32.to_ne_bytes()).unwrap();
        std::fs::write(format!("{}.h3i", hmm.display()), b"index").unwrap();
        std::fs::write(format!("{}.h3m", hmm.display()), b"models").unwrap();

        let err = pressed_db_available(&hmm).unwrap_err();
        assert!(err.to_string().contains("Bad .h3f magic"));
    }

    #[cfg(unix)]
    #[test]
    fn pressed_sidecar_paths_preserve_non_utf8_database_path() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let mut raw = b"/tmp/nonutf8-hmm-".to_vec();
        raw.push(0xff);
        raw.extend_from_slice(b".hmm");
        let hmm_path = PathBuf::from(std::ffi::OsString::from_vec(raw.clone()));

        let paths = pressed_sidecar_paths(&hmm_path);
        for (path, suffix) in paths.iter().zip([".h3f", ".h3p", ".h3i", ".h3m"]) {
            let mut expected = raw.clone();
            expected.extend_from_slice(suffix.as_bytes());
            assert_eq!(path.as_os_str().as_bytes(), expected.as_slice());
        }
    }

    #[test]
    fn pressed_db_available_rejects_mismatched_h3f_h3p_records() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let mut mismatched = hmm.clone();
        mismatched.name = "stale-profile".to_string();
        let om = oprofile_for_press(&mismatched);
        let mut h3p = std::fs::File::create(format!("{}.h3p", hmm_path.display())).unwrap();
        write_h3p_record(&mut h3p, &mismatched, &om).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err
            .to_string()
            .contains("Pressed sidecar first-record mismatch"));
    }

    #[test]
    fn pressed_db_available_rejects_mismatched_h3m_record() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let mut mismatched = hmm.clone();
        mismatched.name = "stale-model".to_string();
        let mut h3m = std::fs::File::create(format!("{}.h3m", hmm_path.display())).unwrap();
        write_binary_hmm(&mut h3m, &mismatched).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err
            .to_string()
            .contains("key fn3 points to HMM stale-model"));
    }

    #[test]
    fn pressed_db_available_accepts_h3i_file_record_from_c_hmmpress() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let wrong_h3m = dir.path().join("other.h3m");
        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        ssi::write_hmm_ssi_records(
            &wrong_h3m,
            &h3i,
            [(hmm.name.clone(), hmm.acc.clone(), 0)],
            true,
        )
        .unwrap();

        assert!(pressed_db_available(&hmm_path).unwrap());
    }

    #[test]
    fn pressed_db_available_rejects_h3i_stale_offset() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        ssi::write_hmm_ssi_records(&h3m, &h3i, [(hmm.name.clone(), hmm.acc.clone(), 1)], true)
            .unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("Bad binary HMM magic"));
    }

    #[test]
    fn pressed_db_available_rejects_h3i_stale_primary_name() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        ssi::write_hmm_ssi_records(&h3m, &h3i, [("stale".to_string(), None, 0)], true).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("key stale points to HMM"));
    }

    #[test]
    fn pressed_db_available_rejects_h3i_primary_data_offset() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let mut bytes = std::fs::read(&h3i).unwrap();
        let data_offset_pos = h3i_primary_data_offset_position(&bytes);
        write_be_u64_at(&mut bytes, data_offset_pos, 1);
        std::fs::write(&h3i, bytes).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("unsupported data_offset=1"));
    }

    #[test]
    fn pressed_db_available_rejects_h3i_primary_record_len() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let mut bytes = std::fs::read(&h3i).unwrap();
        let record_len_pos = h3i_primary_data_offset_position(&bytes) + std::mem::size_of::<u64>();
        write_be_i64_at(&mut bytes, record_len_pos, 7);
        std::fs::write(&h3i, bytes).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("record_len=7"));
    }

    #[test]
    fn pressed_db_available_rejects_duplicate_h3i_secondary_key() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("two.hmm");
        let first = fn3_hmm();
        let mut second = first.clone();
        second.name = "fn3-other".to_string();
        second.acc = Some("PF99999.1".to_string());
        write_two_record_pressed_set(&hmm_path, &first, &second);

        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let mut bytes = std::fs::read(&h3i).unwrap();
        let srecsize = read_be_u32_at(&bytes, 50) as usize;
        let soffset = read_be_u64_at(&bytes, 70) as usize;
        let first_secondary = bytes[soffset..soffset + srecsize].to_vec();
        bytes[soffset + srecsize..soffset + 2 * srecsize].copy_from_slice(&first_secondary);
        std::fs::write(&h3i, bytes).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("duplicate secondary key"));
    }

    #[test]
    fn pressed_db_available_rejects_unsorted_h3i_primary_keys() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("two.hmm");
        let first = fn3_hmm();
        let mut second = first.clone();
        second.name = "fn3-other".to_string();
        second.acc = Some("PF99999.1".to_string());
        write_two_record_pressed_set(&hmm_path, &first, &second);

        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let mut bytes = std::fs::read(&h3i).unwrap();
        let precsize = read_be_u32_at(&bytes, 46) as usize;
        let poffset = read_be_u64_at(&bytes, 62) as usize;
        let first_primary = bytes[poffset..poffset + precsize].to_vec();
        let second_primary = bytes[poffset + precsize..poffset + 2 * precsize].to_vec();
        bytes[poffset..poffset + precsize].copy_from_slice(&second_primary);
        bytes[poffset + precsize..poffset + 2 * precsize].copy_from_slice(&first_primary);
        std::fs::write(&h3i, bytes).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err
            .to_string()
            .contains("primary keys are not strictly sorted"));
    }

    #[test]
    fn pressed_db_available_rejects_unsorted_h3i_secondary_keys() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("two.hmm");
        let first = fn3_hmm();
        let mut second = first.clone();
        second.name = "fn3-other".to_string();
        second.acc = Some("PF99999.1".to_string());
        write_two_record_pressed_set(&hmm_path, &first, &second);

        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let mut bytes = std::fs::read(&h3i).unwrap();
        let srecsize = read_be_u32_at(&bytes, 50) as usize;
        let soffset = read_be_u64_at(&bytes, 70) as usize;
        let first_secondary = bytes[soffset..soffset + srecsize].to_vec();
        let second_secondary = bytes[soffset + srecsize..soffset + 2 * srecsize].to_vec();
        bytes[soffset..soffset + srecsize].copy_from_slice(&second_secondary);
        bytes[soffset + srecsize..soffset + 2 * srecsize].copy_from_slice(&first_secondary);
        std::fs::write(&h3i, bytes).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err
            .to_string()
            .contains("secondary keys are not strictly sorted"));
    }

    #[test]
    fn pressed_db_available_rejects_empty_h3i_secondary_key() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let mut hmm = fn3_hmm();
        hmm.acc = None;
        write_valid_pressed_set(&hmm_path, &hmm);

        let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
        let mut bytes = std::fs::read(&h3i).unwrap();
        let plen = read_be_u32_at(&bytes, 34) as usize;
        let soffset = read_be_u64_at(&bytes, 70) as usize;
        write_be_u64_at(&mut bytes, 22, 1);
        if bytes.len() < soffset + plen {
            bytes.resize(soffset + plen, 0);
        }
        write_fixed_string_at(&mut bytes, soffset, plen, &hmm.name);
        std::fs::write(&h3i, bytes).unwrap();

        let err = pressed_db_available(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("empty secondary key"));
    }

    #[test]
    fn pressed_records_reconstruct_oprofile_exactly() {
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);

        let mut h3f = Vec::new();
        write_h3f_record(&mut h3f, &hmm, &expected, [11, 22, 33]).unwrap();
        let mut h3p = Vec::new();
        write_h3p_record(&mut h3p, &hmm, &expected).unwrap();

        let msv = read_h3f_record(&mut Cursor::new(h3f)).unwrap().unwrap();
        let profile = read_h3p_record(&mut Cursor::new(h3p)).unwrap().unwrap();
        let actual = oprofile_from_pressed_records(&msv, &profile).unwrap();

        assert_eq!(msv.offsets, [11, 22, 33]);
        assert_oprofile_exact(&actual, &expected);
    }

    // === Regression tests for pressed-DB byte parity with C hmmpress ===
    // F1: .h3f field 6 must be om.max_length (HMM MAXL), not the configured L.
    // F2: .h3p annotation arrays must carry a '\0' (not ' ') sentinel at M+1.
    // F3: .h3f compo tail [K..20] must be 0.0 (not COMPO_UNSET) for nucleic.

    /// Parse the `max_length` field and the 20 compo floats out of a written
    /// `.h3f` record (single record). Returns (max_length, compo).
    fn parse_h3f_max_length_and_compo(bytes: &[u8]) -> (i32, [f32; 20]) {
        let mut off = 0usize;
        let rd_i32 = |b: &[u8], o: &mut usize| {
            let v = i32::from_ne_bytes(b[*o..*o + 4].try_into().unwrap());
            *o += 4;
            v
        };
        // magic, M, abc
        let _magic = rd_i32(bytes, &mut off);
        let _m = rd_i32(bytes, &mut off);
        let _abc = rd_i32(bytes, &mut off);
        // name: len-prefixed cstr (len, bytes, NUL)
        let nlen = rd_i32(bytes, &mut off) as usize;
        off += nlen + 1;
        let max_length = rd_i32(bytes, &mut off);
        // compo is the last 20 floats before the 4-byte footer magic.
        let compo_start = bytes.len() - 4 - 20 * 4;
        let mut compo = [0.0f32; 20];
        for (i, c) in compo.iter_mut().enumerate() {
            let o = compo_start + i * 4;
            *c = f32::from_ne_bytes(bytes[o..o + 4].try_into().unwrap());
        }
        (max_length, compo)
    }

    #[test]
    fn h3f_writes_hmm_max_length_not_configured_length() {
        // F1: MADE1 has MAXL 426; configured L is 400. C writes max_length(426).
        let hmm = made1_hmm();
        assert_eq!(hmm.max_length, 426);
        let om = oprofile_for_press(&hmm);
        assert_eq!(om.l, 400, "press configures L=400");
        let mut h3f = Vec::new();
        write_h3f_record(&mut h3f, &hmm, &om, [0, 0, 0]).unwrap();
        let (max_length, _compo) = parse_h3f_max_length_and_compo(&h3f);
        assert_eq!(max_length, 426, "F1: .h3f must store HMM MAXL, not L=400");

        // fn3 (amino, no MAXL) → C writes -1.
        let amino = fn3_hmm();
        assert_eq!(amino.max_length, -1);
        let om_a = oprofile_for_press(&amino);
        let mut h3f_a = Vec::new();
        write_h3f_record(&mut h3f_a, &amino, &om_a, [0, 0, 0]).unwrap();
        let (max_length_a, _) = parse_h3f_max_length_and_compo(&h3f_a);
        assert_eq!(max_length_a, -1, "F1: amino model with no MAXL → -1");
    }

    #[test]
    fn h3f_zero_fills_compo_tail_for_nucleic() {
        // F3: nucleic K=4; compo[4..20] must be 0.0 (C zeroed array), not -1.0.
        let hmm = made1_hmm();
        let om = oprofile_for_press(&hmm);
        let mut h3f = Vec::new();
        write_h3f_record(&mut h3f, &hmm, &om, [0, 0, 0]).unwrap();
        let (_max_length, compo) = parse_h3f_max_length_and_compo(&h3f);
        for (i, &c) in compo.iter().enumerate().skip(4) {
            assert_eq!(c, 0.0, "F3: compo[{i}] tail must be 0.0 for nucleic");
        }
        // [0..4) carry the real composition (finite, not COMPO_UNSET).
        for (i, &c) in compo.iter().enumerate().take(4) {
            assert!(c.is_finite() && c != -1.0, "compo[{i}] should be real");
        }
    }

    #[test]
    fn h3p_annotation_arrays_end_with_nul_sentinel() {
        // F2: each present annotation array of width M+2 must have '\0' at M+1.
        let hmm = fn3_hmm();
        let m = hmm.m;
        let om = oprofile_for_press(&hmm);
        let mut h3p = Vec::new();
        write_h3p_record(&mut h3p, &hmm, &om).unwrap();

        // Re-walk the record header to locate the first annotation array.
        let mut off = 0usize;
        let rd_i32 = |b: &[u8], o: &mut usize| {
            let v = i32::from_ne_bytes(b[*o..*o + 4].try_into().unwrap());
            *o += 4;
            v
        };
        let _magic = rd_i32(&h3p, &mut off);
        let _m = rd_i32(&h3p, &mut off);
        let _abc = rd_i32(&h3p, &mut off);
        let nlen = rd_i32(&h3p, &mut off) as usize; // name
        off += nlen + 1;
        let acclen = rd_i32(&h3p, &mut off) as usize; // acc
        if acclen > 0 {
            off += acclen + 1;
        }
        let desclen = rd_i32(&h3p, &mut off) as usize; // desc
        if desclen > 0 {
            off += desclen + 1;
        }
        // Now four annotation arrays of width m+2: rf, mm, cs, consensus.
        let width = m + 2;
        for arr in 0..4 {
            let base = off + arr * width;
            assert_eq!(
                h3p[base + m + 1],
                0,
                "F2: annotation array {arr} must end with NUL at index M+1"
            );
        }
    }

    #[test]
    fn read_pressed_oprofiles_preserves_h3m_record_order_profile() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);
        write_valid_pressed_set(&hmm_path, &hmm);

        let profiles = read_pressed_oprofiles(&hmm_path).unwrap();

        assert_eq!(profiles.len(), 1);
        assert_oprofile_exact(&profiles[0], &expected);
    }

    #[test]
    fn read_pressed_oprofiles_uses_h3f_profile_offset() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);

        let mut m = std::fs::File::create(&h3m).unwrap();
        write_binary_hmm(&mut m, &hmm).unwrap();
        let mut p = std::fs::File::create(&h3p).unwrap();
        p.write_all(b"prefix").unwrap();
        let profile_offset = p.stream_position().unwrap() as i64;
        write_h3p_record(&mut p, &hmm, &expected).unwrap();

        let mut f = std::fs::File::create(&h3f).unwrap();
        write_h3f_record(&mut f, &hmm, &expected, [0, 0, profile_offset]).unwrap();

        let profiles = read_pressed_oprofiles(&hmm_path).unwrap();

        assert_eq!(profiles.len(), 1);
        assert_oprofile_exact(&profiles[0], &expected);
    }

    #[test]
    fn read_pressed_oprofiles_rejects_negative_h3p_offset() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);

        let mut m = std::fs::File::create(&h3m).unwrap();
        write_binary_hmm(&mut m, &hmm).unwrap();
        let mut p = std::fs::File::create(&h3p).unwrap();
        write_h3p_record(&mut p, &hmm, &expected).unwrap();
        let mut f = std::fs::File::create(&h3f).unwrap();
        write_h3f_record(&mut f, &hmm, &expected, [0, 0, -1]).unwrap();

        let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("negative .h3p offset"));
    }

    #[test]
    fn read_pressed_oprofiles_rejects_extra_h3p_records() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let hmm = fn3_hmm();
        write_valid_pressed_set(&hmm_path, &hmm);

        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let original = std::fs::read(&h3p).unwrap();
        let mut appended = original.clone();
        appended.extend_from_slice(&original);
        std::fs::write(&h3p, appended).unwrap();

        let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn read_pressed_oprofiles_rejects_stale_h3f_foffset() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);

        let mut m = std::fs::File::create(&h3m).unwrap();
        write_binary_hmm(&mut m, &hmm).unwrap();
        let mut p = std::fs::File::create(&h3p).unwrap();
        write_h3p_record(&mut p, &hmm, &expected).unwrap();
        let mut f = std::fs::File::create(&h3f).unwrap();
        write_h3f_record(&mut f, &hmm, &expected, [0, 1, 0]).unwrap();

        let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("FOFFSET mismatch"));
    }

    #[test]
    fn read_pressed_oprofiles_rejects_stale_h3f_moffset() {
        let dir = tempfile::tempdir().unwrap();
        let hmm_path = dir.path().join("fn3.hmm");
        let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
        let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
        let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);

        let mut m = std::fs::File::create(&h3m).unwrap();
        write_binary_hmm(&mut m, &hmm).unwrap();
        let mut p = std::fs::File::create(&h3p).unwrap();
        write_h3p_record(&mut p, &hmm, &expected).unwrap();
        let mut f = std::fs::File::create(&h3f).unwrap();
        write_h3f_record(&mut f, &hmm, &expected, [1, 0, 0]).unwrap();

        let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
        assert!(err.to_string().contains("MOFFSET mismatch"));
    }

    #[test]
    fn validates_pressed_oprofile_order_against_h3m_records() {
        let hmm_a = fn3_hmm();
        let mut hmm_b = hmm_a.clone();
        hmm_b.name = "fn3_alt".to_string();

        let om_a = oprofile_for_press(&hmm_a);
        let om_b = oprofile_for_press(&hmm_b);

        validate_pressed_oprofiles_match_hmms(&[hmm_a.clone(), hmm_b.clone()], &[om_a, om_b])
            .unwrap();

        let err = validate_pressed_oprofiles_match_hmms(
            &[hmm_a, hmm_b],
            &[
                oprofile_for_press(&fn3_hmm()),
                oprofile_for_press(&fn3_hmm()),
            ],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("does not match .h3m record 2"),
            "{err}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn reconstructed_oprofile_scores_like_converted_profile() {
        if !std::is_x86_feature_detected!("sse2") {
            return;
        }

        let hmm = fn3_hmm();
        let mut expected = oprofile_for_press(&hmm);

        let mut h3f = Vec::new();
        write_h3f_record(&mut h3f, &hmm, &expected, [11, 22, 33]).unwrap();
        let mut h3p = Vec::new();
        write_h3p_record(&mut h3p, &hmm, &expected).unwrap();

        let msv = read_h3f_record(&mut Cursor::new(h3f)).unwrap().unwrap();
        let profile = read_h3p_record(&mut Cursor::new(h3p)).unwrap().unwrap();
        let mut actual = oprofile_from_pressed_records(&msv, &profile).unwrap();

        let abc = Alphabet::amino();
        let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY");
        let l = dsq.len() - 2;
        expected.reconfig_length(l as i32);
        actual.reconfig_length(l as i32);

        unsafe {
            assert_eq!(
                msv_score(crate::simd::msv_filter::msv_filter(&dsq, l, &actual)),
                msv_score(crate::simd::msv_filter::msv_filter(&dsq, l, &expected))
            );
            assert_eq!(
                vit_score(crate::simd::vit_filter::viterbi_filter(&dsq, l, &actual)),
                vit_score(crate::simd::vit_filter::viterbi_filter(&dsq, l, &expected))
            );
            assert_eq!(
                crate::simd::fwd_filter::forward_parser(&dsq, l, &actual).to_bits(),
                crate::simd::fwd_filter::forward_parser(&dsq, l, &expected).to_bits()
            );
        }
    }

    #[test]
    fn pressed_records_reconstruction_rejects_stale_pairing() {
        let hmm = fn3_hmm();
        let expected = oprofile_for_press(&hmm);

        let mut h3f = Vec::new();
        write_h3f_record(&mut h3f, &hmm, &expected, [0, 0, 0]).unwrap();
        let mut h3p = Vec::new();
        let mut stale = hmm.clone();
        stale.name = "stale".to_string();
        let stale_om = oprofile_for_press(&stale);
        write_h3p_record(&mut h3p, &stale, &stale_om).unwrap();

        let msv = read_h3f_record(&mut Cursor::new(h3f)).unwrap().unwrap();
        let profile = read_h3p_record(&mut Cursor::new(h3p)).unwrap().unwrap();
        let err = oprofile_from_pressed_records(&msv, &profile).unwrap_err();

        assert!(err.to_string().contains("Pressed profile record mismatch"));
    }
}

/// True iff all four pressed-database sidecars (`.h3f`, `.h3p`, `.h3i`, `.h3m`)
/// exist next to `hmm_path`.
pub fn pressed_db_exists(hmm_path: &Path) -> bool {
    pressed_sidecar_paths(hmm_path).iter().all(|p| p.exists())
}

/// Return `Ok(true)` when every pressed-database sidecar exists, `Ok(false)`
/// when none exist, and an error for partial sidecar sets.
pub fn pressed_db_sidecars_complete(hmm_path: &Path) -> HmmerResult<bool> {
    let paths = pressed_sidecar_paths(hmm_path);
    let existing = paths.iter().filter(|p| p.exists()).count();
    if existing == 0 {
        return Ok(false);
    }
    if existing != paths.len() {
        return Err(HmmerError::Format(format!(
            "Incomplete pressed database for {}: expected .h3f/.h3p/.h3i/.h3m sidecars",
            hmm_path.display()
        )));
    }
    Ok(true)
}

pub fn pressed_h3m_path(hmm_path: &Path) -> PathBuf {
    pressed_sidecar_paths(hmm_path)[3].clone()
}

/// Return `Ok(true)` only when a complete, minimally readable pressed database
/// exists. Return an error for partial or malformed sidecar sets, because C
/// HMMER treats those as pressed-database failures rather than silently falling
/// back to the source HMM.
pub fn pressed_db_available(hmm_path: &Path) -> HmmerResult<bool> {
    if !pressed_db_sidecars_complete(hmm_path)? {
        return Ok(false);
    }
    let paths = pressed_sidecar_paths(hmm_path);

    let mut h3f = std::fs::File::open(&paths[0]).map_err(HmmerError::Io)?;
    let Some(msv) = read_h3f_record(&mut h3f)? else {
        return Err(HmmerError::Format(format!(
            "Pressed MSV sidecar {} contains no records",
            paths[0].display()
        )));
    };
    let mut h3p = std::fs::File::open(&paths[1]).map_err(HmmerError::Io)?;
    let Some(profile) = read_h3p_record(&mut h3p)? else {
        return Err(HmmerError::Format(format!(
            "Pressed profile sidecar {} contains no records",
            paths[1].display()
        )));
    };
    if (msv.name.as_str(), msv.m, msv.abc_type)
        != (profile.name.as_str(), profile.m, profile.abc_type)
    {
        return Err(HmmerError::Format(format!(
            "Pressed sidecar first-record mismatch: .h3f has {} M={} alphabet={}, .h3p has {} M={} alphabet={}",
            msv.name, msv.m, msv.abc_type, profile.name, profile.m, profile.abc_type
        )));
    }

    validate_h3i_against_h3m(&paths[2], &paths[3])?;
    let h3m = std::fs::File::open(&paths[3]).map_err(HmmerError::Io)?;
    let mut h3m = BufReader::new(h3m);
    let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut h3m)? else {
        return Err(HmmerError::Format(format!(
            "Pressed binary model sidecar {} contains no records",
            paths[3].display()
        )));
    };
    let h3m_abc = alphabet_type_code(hmm.abc_type)?;
    if (msv.name.as_str(), msv.m, msv.abc_type) != (hmm.name.as_str(), hmm.m, h3m_abc) {
        return Err(HmmerError::Format(format!(
            "Pressed sidecar/model mismatch: sidecars have {} M={} alphabet={}, .h3m has {} M={} alphabet={}",
            msv.name, msv.m, msv.abc_type, hmm.name, hmm.m, h3m_abc
        )));
    }

    let hmms = hmmfile_binary::read_binary_hmm_file(&paths[3])?;
    let oprofiles = read_pressed_oprofiles(hmm_path)?;
    validate_pressed_oprofiles_match_hmms(&hmms, &oprofiles)?;

    Ok(true)
}

/// Read all optimized profiles from a complete C-style pressed database.
///
/// Profiles are returned in the same order as their corresponding `.h3m`
/// records. Scans still need the `.h3m` HMMs for generic profile
/// configuration, thresholds, alignment, and domain definition.
pub fn read_pressed_oprofiles(hmm_path: &Path) -> HmmerResult<Vec<OProfile>> {
    let paths = pressed_sidecar_paths(hmm_path);
    let mut h3f = std::fs::File::open(&paths[0]).map_err(HmmerError::Io)?;
    let mut h3p = std::fs::File::open(&paths[1]).map_err(HmmerError::Io)?;
    let h3m = std::fs::File::open(&paths[3]).map_err(HmmerError::Io)?;
    let mut h3m = BufReader::new(h3m);

    let mut profiles = Vec::new();
    let mut max_h3p_position = 0_u64;
    loop {
        let current_foffset = h3f.stream_position().map_err(HmmerError::Io)? as i64;
        let current_moffset = h3m.stream_position().map_err(HmmerError::Io)? as i64;
        match read_h3f_record(&mut h3f)? {
            Some(msv) => {
                if msv.offsets[1] != current_foffset {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f FOFFSET mismatch for {}: record starts at {}, offset says {}",
                        msv.name, current_foffset, msv.offsets[1]
                    )));
                }
                if msv.offsets[0] != current_moffset {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f MOFFSET mismatch for {}: .h3m record starts at {}, offset says {}",
                        msv.name, current_moffset, msv.offsets[0]
                    )));
                }
                let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut h3m)? else {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f record {} has no matching .h3m record",
                        msv.name
                    )));
                };
                if msv.name != hmm.name {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f record {} points to .h3m record {}",
                        msv.name, hmm.name
                    )));
                }
                let h3m_abc = alphabet_type_code(hmm.abc_type)?;
                if msv.abc_type != h3m_abc {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f record {} alphabet {} does not match .h3m alphabet {}",
                        msv.name, msv.abc_type, h3m_abc
                    )));
                }
                let p_offset = msv.offsets[2];
                if p_offset < 0 {
                    return Err(HmmerError::Format(format!(
                        "Pressed database has negative .h3p offset for {}: {}",
                        msv.name, p_offset
                    )));
                }
                h3p.seek(SeekFrom::Start(p_offset as u64))
                    .map_err(HmmerError::Io)?;
                let Some(profile) = read_h3p_record(&mut h3p)? else {
                    return Err(HmmerError::Format(format!(
                        "Pressed database has .h3f record {} without matching .h3p record",
                        msv.name
                    )));
                };
                max_h3p_position =
                    max_h3p_position.max(h3p.stream_position().map_err(HmmerError::Io)?);
                profiles.push(oprofile_from_pressed_records(&msv, &profile)?);
            }
            None => break,
        }
    }
    let h3p_len = h3p.metadata().map_err(HmmerError::Io)?.len();
    if max_h3p_position != h3p_len {
        return Err(HmmerError::Format(format!(
            "Pressed .h3p sidecar has {} trailing bytes after matched records",
            h3p_len.saturating_sub(max_h3p_position)
        )));
    }

    Ok(profiles)
}

/// Validate that hydrated pressed optimized profiles line up with `.h3m`
/// HMM records in count and order before scan code indexes them in parallel.
pub fn validate_pressed_oprofiles_match_hmms(
    hmms: &[Hmm],
    oprofiles: &[OProfile],
) -> HmmerResult<()> {
    if oprofiles.len() != hmms.len() {
        return Err(HmmerError::Format(format!(
            ".h3f/.h3p contain {} records but .h3m contains {}",
            oprofiles.len(),
            hmms.len()
        )));
    }

    for (idx, (hmm, om)) in hmms.iter().zip(oprofiles.iter()).enumerate() {
        let expected_abc = alphabet_dimensions(alphabet_type_code(hmm.abc_type)?, ".h3m")?;
        if hmm.name != om.name
            || hmm.m != om.m
            || hmm.abc_k != om.abc_k
            || expected_abc.1 != om.abc_kp
        {
            return Err(HmmerError::Format(format!(
                "Pressed optimized profile {} does not match .h3m record {}: .h3m has {} M={} K={} Kp={}, .h3f/.h3p has {} M={} K={} Kp={}",
                idx + 1,
                idx + 1,
                hmm.name,
                hmm.m,
                hmm.abc_k,
                expected_abc.1,
                om.name,
                om.m,
                om.abc_k,
                om.abc_kp
            )));
        }
    }

    Ok(())
}

fn validate_h3i_against_h3m(path: &Path, h3m_path: &Path) -> HmmerResult<()> {
    let mut file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let magic = read_u32_be(&mut file).map_err(|e| {
        if matches!(e, HmmerError::Io(ref io) if io.kind() == ErrorKind::UnexpectedEof) {
            HmmerError::Format(format!(
                "Pressed SSI sidecar {} is truncated",
                path.display()
            ))
        } else {
            e
        }
    })?;
    if magic != SSI_V30_MAGIC {
        return Err(HmmerError::Format(format!(
            "Bad .h3i SSI magic in {}: {:#x}",
            path.display(),
            magic
        )));
    }
    let _flags = read_u32_be(&mut file)?;
    let offsz = read_u32_be(&mut file)?;
    let nfiles = read_u16_be(&mut file)?;
    let nprimary = read_u64_be(&mut file)?;
    let nsecondary = read_u64_be(&mut file)?;
    let flen = read_u32_be(&mut file)? as usize;
    let plen = read_u32_be(&mut file)? as usize;
    let slen = read_u32_be(&mut file)? as usize;
    let frecsize = read_u32_be(&mut file)? as usize;
    let precsize = read_u32_be(&mut file)? as usize;
    let srecsize = read_u32_be(&mut file)? as usize;
    let foffset = read_u64_be(&mut file)?;
    let poffset = read_u64_be(&mut file)?;
    let soffset = read_u64_be(&mut file)?;

    if offsz != 8 || nfiles != 1 {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} has unsupported header: offsz={} nfiles={}",
            path.display(),
            offsz,
            nfiles
        )));
    }
    let min_frecsize = flen + 4 * std::mem::size_of::<u32>();
    let min_precsize = plen + std::mem::size_of::<u16>() + 2 * 8 + 8;
    let min_srecsize = slen + plen;
    if frecsize != min_frecsize || precsize != min_precsize || srecsize != min_srecsize {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} has inconsistent record sizes",
            path.display()
        )));
    }

    file.seek(SeekFrom::Start(foffset))
        .map_err(HmmerError::Io)?;
    let _indexed_file = read_fixed_string(&mut file, flen)?;
    skip_exact(&mut file, 4 * std::mem::size_of::<u32>())?;

    let h3m_len = std::fs::metadata(h3m_path).map_err(HmmerError::Io)?.len();
    let mut h3m = std::fs::File::open(h3m_path).map_err(HmmerError::Io)?;
    let mut primary_names = HashSet::new();
    let mut primary_acc_by_name = HashMap::new();
    let mut last_primary_key: Option<String> = None;
    file.seek(SeekFrom::Start(poffset))
        .map_err(HmmerError::Io)?;
    for _ in 0..nprimary {
        let key = read_fixed_string(&mut file, plen)?;
        if key.is_empty() {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains empty primary key",
                path.display()
            )));
        }
        if !primary_names.insert(key.clone()) {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains duplicate primary key {}",
                path.display(),
                key
            )));
        }
        if last_primary_key
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} primary keys are not strictly sorted",
                path.display()
            )));
        }
        last_primary_key = Some(key.clone());
        let file_idx = read_u16_be(&mut file)?;
        let offset = read_u64_be(&mut file)?;
        let data_offset = read_u64_be(&mut file)?;
        let record_len = read_i64_be(&mut file)?;
        if data_offset != 0 || record_len != 0 {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} primary key {} has unsupported data_offset={} record_len={}",
                path.display(),
                key,
                data_offset,
                record_len
            )));
        }
        if file_idx != 0 || offset >= h3m_len {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} has invalid offset for key {}",
                path.display(),
                key
            )));
        }
        h3m.seek(SeekFrom::Start(offset)).map_err(HmmerError::Io)?;
        let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut h3m)? else {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} offset for key {} does not point to an HMM",
                path.display(),
                key
            )));
        };
        if hmm.name != key {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} key {} points to HMM {}",
                path.display(),
                key,
                hmm.name
            )));
        }
        primary_acc_by_name.insert(key, hmm.acc.unwrap_or_default());
    }

    let mut h3m_scan = BufReader::new(std::fs::File::open(h3m_path).map_err(HmmerError::Io)?);
    let mut h3m_count = 0u64;
    while hmmfile_binary::read_binary_hmm(&mut h3m_scan)?.is_some() {
        h3m_count += 1;
    }
    if h3m_count != nprimary {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} indexes {} primary keys but .h3m contains {} records",
            path.display(),
            nprimary,
            h3m_count
        )));
    }

    file.seek(SeekFrom::Start(soffset))
        .map_err(HmmerError::Io)?;
    let mut secondary_keys = HashSet::new();
    let mut last_secondary_key: Option<String> = None;
    for _ in 0..nsecondary {
        let acc = read_fixed_string(&mut file, slen)?;
        let primary = read_fixed_string(&mut file, plen)?;
        if acc.is_empty() {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains empty secondary key",
                path.display()
            )));
        }
        if !secondary_keys.insert(acc.clone()) {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains duplicate secondary key {}",
                path.display(),
                acc
            )));
        }
        if last_secondary_key
            .as_ref()
            .is_some_and(|previous| previous >= &acc)
        {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} secondary keys are not strictly sorted",
                path.display()
            )));
        }
        last_secondary_key = Some(acc.clone());
        let Some(expected_acc) = primary_acc_by_name.get(&primary) else {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} secondary key {} references missing primary {}",
                path.display(),
                acc,
                primary
            )));
        };
        if !primary_names.contains(&primary) || expected_acc != &acc {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} secondary key {} does not match primary {} accession",
                path.display(),
                acc,
                primary
            )));
        }
    }
    Ok(())
}

fn read_u16_be<R: Read>(r: &mut R) -> HmmerResult<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32_be<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64_be<R: Read>(r: &mut R) -> HmmerResult<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_i64_be<R: Read>(r: &mut R) -> HmmerResult<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i64::from_be_bytes(buf))
}

fn read_fixed_string<R: Read>(r: &mut R, len: usize) -> HmmerResult<String> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..end]).to_string())
}

fn skip_exact<R: Read>(r: &mut R, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)
}

fn pressed_sidecar_paths(hmm_path: &Path) -> [PathBuf; 4] {
    let sidecar_path = |suffix: &str| {
        let mut path = hmm_path.as_os_str().to_os_string();
        path.push(suffix);
        PathBuf::from(path)
    };
    [
        sidecar_path(".h3f"),
        sidecar_path(".h3p"),
        sidecar_path(".h3i"),
        sidecar_path(".h3m"),
    ]
}

/// Viterbi/Forward profile data from .h3p file for one profile.
pub struct ProfileData {
    pub name: String,
    pub acc: String,
    pub desc: String,
    pub m: usize,
    pub abc_type: i32,
    /// Raw Viterbi transition/emission data
    pub viterbi_data: Vec<u8>,
    /// Raw Forward transition/emission data
    pub forward_data: Vec<u8>,
    pub xw: [[i16; 2]; 4],
    pub xf: [[f32; 2]; 4],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub ncj_roundoff: f32,
    pub cutoff: [f32; 6],
    pub nj: f32,
    pub mode: i32,
    pub l: i32,
}

/// Read one Viterbi/Forward profile (`.h3p`) record produced by `hmmpress`.
/// Reads name/acc/desc, annotation strings, the striped Viterbi (16-bit) and
/// Forward (float) parameter blocks, and trailing cutoffs/metadata. Mirrors
/// `p7_oprofile_ReadRest` in `hmmer/src/p7_hmmfile.c`. Returns `Ok(None)` at EOF.
pub fn read_h3p_record<R: Read>(r: &mut R) -> HmmerResult<Option<ProfileData>> {
    let Some(magic) = read_record_magic_ne(r)? else {
        return Ok(None);
    };
    if magic != V3F_PMAGIC {
        return Err(HmmerError::Format(format!("Bad .h3p magic: {:#x}", magic)));
    }

    let m = read_i32_ne(r)?;
    let abc_type = read_i32_ne(r)?;
    let m = validate_pressed_record_header(m, abc_type, ".h3p")?;

    // Name
    let name_len = read_nonnegative_len(r, ".h3p name length")?;
    let name = require_nonempty_required_string(
        read_len_prefixed_cstr(r, name_len, ".h3p name")?,
        ".h3p name",
    )?;

    // Accession (optional, length-prefixed)
    let acc_len = read_nonnegative_len(r, ".h3p accession length")?;
    let acc = read_len_prefixed_cstr(r, acc_len, ".h3p accession")?;

    // Description
    let desc_len = read_nonnegative_len(r, ".h3p description length")?;
    let desc = read_len_prefixed_cstr(r, desc_len, ".h3p description")?;

    // RF, MM, CS, consensus strings
    for _ in 0..4 {
        read_bytes(r, m + 2)?; // each is M+2 bytes
    }

    // Viterbi data (twv + rwv)
    let (_k, kp) = alphabet_dimensions(abc_type, ".h3p")?;
    let qw = ((m.max(1) - 1) / 8 + 1).max(2);
    let vit_size = (8 * qw * 16) + (kp * qw * 16); // twv + rwv in bytes
    let viterbi_data = read_bytes(r, vit_size)?;

    // Viterbi special states
    let mut xw = [[0i16; 2]; 4];
    for s in 0..4 {
        for t in 0..2 {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf).map_err(HmmerError::Io)?;
            xw[s][t] = i16::from_ne_bytes(buf);
        }
    }
    let scale_w = read_f32_ne(r)?;
    let mut buf2 = [0u8; 2];
    r.read_exact(&mut buf2).map_err(HmmerError::Io)?;
    let base_w = i16::from_ne_bytes(buf2);
    r.read_exact(&mut buf2).map_err(HmmerError::Io)?; // ddbound_w
    let ddbound_w = i16::from_ne_bytes(buf2);
    let ncj_roundoff = read_f32_ne(r)?;

    // Forward data (tfv + rfv)
    let qf = ((m.max(1) - 1) / 4 + 1).max(2);
    let fwd_size = (8 * qf * 16) + (kp * qf * 16);
    let forward_data = read_bytes(r, fwd_size)?;

    // Forward special states
    let mut xf = [[0.0f32; 2]; 4];
    for s in 0..4 {
        for t in 0..2 {
            xf[s][t] = read_f32_ne(r)?;
        }
    }

    // Cutoffs
    let mut cutoff = [0.0f32; 6];
    for i in 0..6 {
        cutoff[i] = read_f32_ne(r)?;
    }

    let nj = read_f32_ne(r)?;
    let mode = read_i32_ne(r)?;
    let l = read_i32_ne(r)?;

    // Footer
    let footer = read_u32_ne(r)?;
    if footer != V3F_PMAGIC {
        return Err(HmmerError::Format("Bad .h3p footer".to_string()));
    }

    Ok(Some(ProfileData {
        name,
        acc,
        desc,
        m,
        abc_type,
        viterbi_data,
        forward_data,
        xw,
        xf,
        scale_w,
        base_w,
        ddbound_w,
        ncj_roundoff,
        cutoff,
        nj,
        mode,
        l,
    }))
}

fn read_i16_vec8(bytes: &[u8], offset: &mut usize, field: &str) -> HmmerResult<[i16; 8]> {
    if bytes.len().saturating_sub(*offset) < 16 {
        return Err(HmmerError::Format(format!(
            "Truncated pressed {field} vector"
        )));
    }
    let mut out = [0i16; 8];
    for value in &mut out {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&bytes[*offset..*offset + 2]);
        *value = i16::from_ne_bytes(buf);
        *offset += 2;
    }
    Ok(out)
}

fn read_f32_vec4(bytes: &[u8], offset: &mut usize, field: &str) -> HmmerResult<[f32; 4]> {
    if bytes.len().saturating_sub(*offset) < 16 {
        return Err(HmmerError::Format(format!(
            "Truncated pressed {field} vector"
        )));
    }
    let mut out = [0.0f32; 4];
    for value in &mut out {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[*offset..*offset + 4]);
        *value = f32::from_ne_bytes(buf);
        *offset += 4;
    }
    Ok(out)
}

/// Reconstruct an optimized profile from matching `.h3f` and `.h3p` records.
///
/// This intentionally does not replace scan's current `.h3m` path yet: scan
/// still needs the generic profile/HMM for thresholding, alignment, and domain
/// work. The constructor gives the acceleration payload a focused parity point.
pub fn oprofile_from_pressed_records(
    msv: &MsvFilterData,
    profile: &ProfileData,
) -> HmmerResult<OProfile> {
    if (msv.name.as_str(), msv.m, msv.abc_type)
        != (profile.name.as_str(), profile.m, profile.abc_type)
    {
        return Err(HmmerError::Format(format!(
            "Pressed profile record mismatch: .h3f has {} M={} alphabet={}, .h3p has {} M={} alphabet={}",
            msv.name, msv.m, msv.abc_type, profile.name, profile.m, profile.abc_type
        )));
    }

    let (abc_k, abc_kp) = alphabet_dimensions(msv.abc_type, "pressed profile")?;
    let q16 = nqb(msv.m);
    let q16x = q16 + P7O_EXTRA_SB;
    let q8 = nqw(msv.m);
    let q4 = nqf(msv.m);

    let expected_msv = abc_kp * (q16x + q16) * 16;
    if msv.msv_scores.len() != expected_msv {
        return Err(HmmerError::Format(format!(
            "Pressed .h3f vector block has {} bytes, expected {}",
            msv.msv_scores.len(),
            expected_msv
        )));
    }
    let expected_vit = (8 * q8 * 16) + (abc_kp * q8 * 16);
    if profile.viterbi_data.len() != expected_vit {
        return Err(HmmerError::Format(format!(
            "Pressed .h3p Viterbi block has {} bytes, expected {}",
            profile.viterbi_data.len(),
            expected_vit
        )));
    }
    let expected_fwd = (8 * q4 * 16) + (abc_kp * q4 * 16);
    if profile.forward_data.len() != expected_fwd {
        return Err(HmmerError::Format(format!(
            "Pressed .h3p Forward block has {} bytes, expected {}",
            profile.forward_data.len(),
            expected_fwd
        )));
    }

    let mut offset = 0;
    let mut sbv = vec![vec![[0u8; 16]; q16x]; abc_kp];
    for rows in &mut sbv {
        for vector in rows {
            vector.copy_from_slice(&msv.msv_scores[offset..offset + 16]);
            offset += 16;
        }
    }
    let mut rbv = vec![vec![[0u8; 16]; q16]; abc_kp];
    for rows in &mut rbv {
        for vector in rows {
            vector.copy_from_slice(&msv.msv_scores[offset..offset + 16]);
            offset += 16;
        }
    }

    let mut offset = 0;
    let mut twv = vec![[0i16; 8]; 8 * q8];
    for vector in &mut twv {
        *vector = read_i16_vec8(&profile.viterbi_data, &mut offset, ".h3p twv")?;
    }
    let mut rwv = vec![vec![[0i16; 8]; q8]; abc_kp];
    for rows in &mut rwv {
        for vector in rows {
            *vector = read_i16_vec8(&profile.viterbi_data, &mut offset, ".h3p rwv")?;
        }
    }

    let mut offset = 0;
    let mut tfv = vec![[0.0f32; 4]; 8 * q4];
    for vector in &mut tfv {
        *vector = read_f32_vec4(&profile.forward_data, &mut offset, ".h3p tfv")?;
    }
    let mut rfv = vec![vec![[0.0f32; 4]; q4]; abc_kp];
    for rows in &mut rfv {
        for vector in rows {
            *vector = read_f32_vec4(&profile.forward_data, &mut offset, ".h3p rfv")?;
        }
    }

    #[cfg(target_arch = "x86_64")]
    let rfv_a: Vec<Vec<AlignedF32x4>> = rfv
        .iter()
        .map(|rows| rows.iter().copied().map(AlignedF32x4::from_array).collect())
        .collect();
    #[cfg(target_arch = "x86_64")]
    let tfv_a: Vec<AlignedF32x4> = tfv.iter().copied().map(AlignedF32x4::from_array).collect();

    Ok(OProfile {
        rbv,
        sbv,
        tbm_b: msv.tbm_b,
        tec_b: msv.tec_b,
        tjb_b: msv.tjb_b,
        scale_b: msv.scale_b,
        base_b: msv.base_b,
        bias_b: msv.bias_b,
        rwv,
        twv,
        xw: profile.xw,
        scale_w: profile.scale_w,
        base_w: profile.base_w,
        ddbound_w: profile.ddbound_w,
        ncj_roundoff: profile.ncj_roundoff,
        rfv,
        tfv,
        #[cfg(target_arch = "x86_64")]
        rfv_a,
        #[cfg(target_arch = "x86_64")]
        tfv_a,
        xf: profile.xf,
        m: msv.m,
        l: profile.l,
        max_length: msv.max_length,
        nj: profile.nj,
        mode: profile.mode,
        evparam: msv.evparam,
        cutoff: profile.cutoff,
        compo: msv.compo,
        name: msv.name.clone(),
        abc_k,
        abc_kp,
    })
}
