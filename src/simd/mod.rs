pub mod oprofile;
#[cfg(target_arch = "x86_64")]
pub mod optacc;

#[cfg(target_arch = "x86_64")]
pub mod avx2_fwd;
#[cfg(target_arch = "x86_64")]
pub mod avx2_msv;
#[cfg(target_arch = "x86_64")]
pub mod avx2_vit;
#[cfg(target_arch = "x86_64")]
pub mod bck_filter;
#[cfg(target_arch = "x86_64")]
pub mod fwd_filter;
#[cfg(target_arch = "x86_64")]
pub mod msv_filter;
pub mod probmx;
#[cfg(target_arch = "x86_64")]
pub mod ssv_filter;
pub mod ssv_longtarget;
#[cfg(target_arch = "x86_64")]
pub mod vit_filter;

pub mod neon_fwd;
pub mod neon_msv;
pub mod neon_vit;

/// Build an 8-bit immediate shuffle mask for `_mm_shuffle_epi32` / `_mm_shufflelo_epi16`.
/// The four 2-bit fields select source lanes (z = high, w = low).
pub const fn shuffle_mask(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
