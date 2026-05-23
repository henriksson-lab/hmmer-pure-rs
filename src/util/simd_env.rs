//! Processor-specific floating point setup matching HMMER's `impl_Init()`.

/// Enable the SSE floating point mode used by C HMMER (`impl_Init`).
///
/// On x86/x86_64, sets MXCSR flush-to-zero (FTZ) and denormals-are-zero (DAZ)
/// so subnormals are treated as zero. No-op on non-x86 targets.
pub fn init() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        set_x86_flush_zero();
    }
}

/// Set the MXCSR FTZ+DAZ bits on x86_64 via inline assembly.
#[cfg(target_arch = "x86_64")]
unsafe fn set_x86_flush_zero() {
    const MXCSR_DAZ: u32 = 1 << 6;
    const MXCSR_FTZ: u32 = 1 << 15;

    // SAFETY: Reading and writing MXCSR is a per-thread processor mode change.
    // The bit mask only enables DAZ/FTZ, matching C HMMER's impl_Init().
    let mut csr = 0u32;
    unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags)) };
    let new_csr = csr | MXCSR_DAZ | MXCSR_FTZ;
    unsafe { std::arch::asm!("ldmxcsr [{}]", in(reg) &new_csr, options(nostack, preserves_flags)) };
}

/// Set the MXCSR FTZ+DAZ bits on 32-bit x86 via inline assembly.
#[cfg(target_arch = "x86")]
unsafe fn set_x86_flush_zero() {
    const MXCSR_DAZ: u32 = 1 << 6;
    const MXCSR_FTZ: u32 = 1 << 15;

    // SAFETY: Reading and writing MXCSR is a per-thread processor mode change.
    // The bit mask only enables DAZ/FTZ, matching C HMMER's impl_Init().
    let mut csr = 0u32;
    unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut csr, options(nostack, preserves_flags)) };
    let new_csr = csr | MXCSR_DAZ | MXCSR_FTZ;
    unsafe { std::arch::asm!("ldmxcsr [{}]", in(reg) &new_csr, options(nostack, preserves_flags)) };
}
