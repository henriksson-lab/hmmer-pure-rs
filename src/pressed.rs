//! Read/write C HMMER pressed database format (.h3f, .h3p, .h3i).
//! Enables reading databases created by C hmmpress.

use std::io::Read;
use std::path::Path;

use crate::errors::{HmmerError, HmmerResult};

// Magic numbers
const V3F_FMAGIC: u32 = 0xe8ededb5; // .h3f sentinel
#[allow(dead_code)]
const V3F_PMAGIC: u32 = 0xe8ededb4; // .h3p sentinel (used for future .h3p reader)

fn read_u32_ne<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_ne_bytes(buf))
}

fn read_i32_ne<R: Read>(r: &mut R) -> HmmerResult<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i32::from_ne_bytes(buf))
}

fn read_f32_ne<R: Read>(r: &mut R) -> HmmerResult<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(f32::from_ne_bytes(buf))
}

fn read_u8<R: Read>(r: &mut R) -> HmmerResult<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf[0])
}

fn read_bytes<R: Read>(r: &mut R, n: usize) -> HmmerResult<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf)
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
    pub compo: [f32; 20],
}

/// Read one MSV filter record from a .h3f file.
pub fn read_h3f_record<R: Read>(r: &mut R) -> HmmerResult<Option<MsvFilterData>> {
    let magic = match read_u32_ne(r) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if magic != V3F_FMAGIC {
        return Err(HmmerError::Format(format!("Bad .h3f magic: {:#x}", magic)));
    }

    let m = read_i32_ne(r)? as usize;
    let abc_type = read_i32_ne(r)?;
    let name_len = read_i32_ne(r)? as usize;
    let name_bytes = read_bytes(r, name_len + 1)?;
    let name = String::from_utf8_lossy(&name_bytes[..name_len]).to_string();
    let max_length = read_i32_ne(r)?;

    let tbm_b = read_u8(r)?;
    let tec_b = read_u8(r)?;
    let tjb_b = read_u8(r)?;
    let scale_b = read_f32_ne(r)?;
    let base_b = read_u8(r)?;
    let bias_b = read_u8(r)?;

    // Read MSV score vectors (raw bytes)
    // Size depends on alphabet and Q
    let k = if abc_type == 3 { 29 } else { 18 }; // Kp
    let q = ((m.max(1) - 1) / 16 + 1).max(2);
    let msv_bytes = k * q * 16 * 2; // sbv + rbv
    let msv_scores = read_bytes(r, msv_bytes)?;

    let mut evparam = [0.0f32; 6];
    for i in 0..6 {
        evparam[i] = read_f32_ne(r)?;
    }

    // offs (3 off_t values)
    for _ in 0..3 {
        read_bytes(r, std::mem::size_of::<i64>())?;
    }

    let mut compo = [0.0f32; 20];
    let compo_k = if abc_type == 3 { 20 } else { 4 };
    for i in 0..compo_k {
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
        compo,
    }))
}

/// Check if a pressed database exists for an HMM file.
pub fn pressed_db_exists(hmm_path: &Path) -> bool {
    let base = hmm_path.to_str().unwrap_or("");
    Path::new(&format!("{}.h3f", base)).exists()
        && Path::new(&format!("{}.h3p", base)).exists()
        && Path::new(&format!("{}.h3i", base)).exists()
        && Path::new(&format!("{}.h3m", base)).exists()
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
    pub cutoff: [f32; 6],
    pub nj: f32,
    pub mode: i32,
    pub l: i32,
}

/// Read one profile record from a .h3p file.
pub fn read_h3p_record<R: Read>(r: &mut R) -> HmmerResult<Option<ProfileData>> {
    let magic = match read_u32_ne(r) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if magic != V3F_PMAGIC {
        return Err(HmmerError::Format(format!("Bad .h3p magic: {:#x}", magic)));
    }

    let m = read_i32_ne(r)? as usize;
    let abc_type = read_i32_ne(r)?;

    // Name
    let name_len = read_i32_ne(r)? as usize;
    let name_bytes = read_bytes(r, name_len + 1)?;
    let name = String::from_utf8_lossy(&name_bytes[..name_len]).to_string();

    // Accession (optional, length-prefixed)
    let acc_len = read_i32_ne(r)? as usize;
    let acc = if acc_len > 0 {
        let b = read_bytes(r, acc_len + 1)?;
        String::from_utf8_lossy(&b[..acc_len]).to_string()
    } else {
        String::new()
    };

    // Description
    let desc_len = read_i32_ne(r)? as usize;
    let desc = if desc_len > 0 {
        let b = read_bytes(r, desc_len + 1)?;
        String::from_utf8_lossy(&b[..desc_len]).to_string()
    } else {
        String::new()
    };

    // RF, MM, CS, consensus strings
    for _ in 0..4 {
        read_bytes(r, m + 2)?; // each is M+2 bytes
    }

    // Viterbi data (twv + rwv)
    let kp = if abc_type == 3 { 29 } else { 18 };
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
    read_f32_ne(r)?; // ncj_roundoff

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
        cutoff,
        nj,
        mode,
        l,
    }))
}
