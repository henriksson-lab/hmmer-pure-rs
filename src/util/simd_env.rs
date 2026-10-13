//! Processor-specific floating point setup matching HMMER's `impl_Init()`.

/// Enable the SSE floating point mode used by C HMMER (`impl_Init`).
///
/// On x86/x86_64, sets MXCSR flush-to-zero (FTZ) and denormals-are-zero (DAZ)
/// so subnormals are treated as zero. On aarch64, sets FPCR.FZ, which flushes
/// subnormal inputs and outputs alike, so the SSE kernels (run there over NEON,
/// see `simd::sse`) round exactly as they do on x86. No-op elsewhere.
pub fn init() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        set_x86_flush_zero();
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: only sets the per-thread FPCR.FZ mode bit; see the function.
    unsafe {
        set_aarch64_flush_zero();
    }
}

/// Set FPCR.FZ (bit 24) on aarch64 via inline assembly.
#[cfg(target_arch = "aarch64")]
unsafe fn set_aarch64_flush_zero() {
    const FPCR_FZ: u64 = 1 << 24;

    // SAFETY: Reading and writing FPCR is a per-thread processor mode change.
    // The bit mask only enables flush-to-zero, the counterpart of x86 FTZ+DAZ.
    let mut fpcr: u64;
    unsafe {
        std::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags))
    };
    fpcr |= FPCR_FZ;
    unsafe {
        std::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nomem, nostack, preserves_flags))
    };
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
