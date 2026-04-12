pub mod oprofile;

#[cfg(target_arch = "x86_64")]
pub mod msv_filter;
#[cfg(target_arch = "x86_64")]
pub mod vit_filter;
#[cfg(target_arch = "x86_64")]
pub mod fwd_filter;
#[cfg(target_arch = "x86_64")]
pub mod bck_filter;
#[cfg(target_arch = "x86_64")]
pub mod avx2_msv;
#[cfg(target_arch = "x86_64")]
pub mod avx2_vit;
#[cfg(target_arch = "x86_64")]
pub mod avx2_fwd;

pub mod neon_msv;
pub mod neon_vit;
pub mod neon_fwd;

/// Helper to create shuffle mask constant for _mm_shuffle_epi32 / _mm_shufflelo_epi16.
pub const fn shuffle_mask(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
