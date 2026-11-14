//! The SSE2 intrinsics the striped kernels are written in, on every SIMD target.
//!
//! On x86_64 this is `std::arch::x86_64`, re-exported. On aarch64 it is the same
//! names and signatures implemented with NEON (in the spirit of C's `sse2neon`),
//! so the one SSE source of each kernel also runs on Apple Silicon and ARM
//! servers, performing the same operations in the same order. That, not speed
//! alone, is why the kernels are shared rather than given separate NEON ports:
//! posterior decoding and domain envelopes are sensitive to every rounding, and
//! a separately written kernel (or the generic log-space DP) lands envelopes a
//! few residues away from the C/x86 results.
//!
//! Only the intrinsics this crate uses are provided, each with SSE's exact lane
//! semantics where NEON's natural instruction differs: `_mm_max_ps` returns its
//! second operand when the comparison is unordered, `_mm_cvttps_epi32` yields
//! `i32::MIN` for NaN and out-of-range lanes, and the byte shifts fill with zero.
//! Float add/sub/mul are single IEEE operations on both (never fused).

#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
pub use neon::*;

/// Whether the SSE kernels can run here: SSE2 is detected on x86_64; NEON is
/// architecturally mandatory on aarch64.
#[inline]
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types, clippy::missing_safety_doc)]
mod neon {
    use std::arch::aarch64::*;

    /// Four `f32` lanes, as SSE's `__m128`.
    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct __m128(pub float32x4_t);

    /// 128 bits viewed as bytes, words or dwords by each operation, as SSE's
    /// `__m128i`.
    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct __m128i(pub uint8x16_t);

    // NEON is baseline on aarch64, so none of these needs a target_feature gate;
    // they are `unsafe fn` only to keep the call sites identical to x86's.

    #[inline(always)]
    fn f(a: __m128) -> float32x4_t {
        a.0
    }
    #[inline(always)]
    fn u32s(a: __m128) -> uint32x4_t {
        // SAFETY: reinterpreting 128 bits between lane types is a no-op.
        unsafe { vreinterpretq_u32_f32(a.0) }
    }
    #[inline(always)]
    fn from_u32s(v: uint32x4_t) -> __m128 {
        // SAFETY: as above.
        __m128(unsafe { vreinterpretq_f32_u32(v) })
    }
    #[inline(always)]
    fn i16s(a: __m128i) -> int16x8_t {
        // SAFETY: as above.
        unsafe { vreinterpretq_s16_u8(a.0) }
    }
    #[inline(always)]
    fn from_i16s(v: int16x8_t) -> __m128i {
        // SAFETY: as above.
        __m128i(unsafe { vreinterpretq_u8_s16(v) })
    }
    #[inline(always)]
    fn i32s(a: __m128i) -> int32x4_t {
        // SAFETY: as above.
        unsafe { vreinterpretq_s32_u8(a.0) }
    }
    #[inline(always)]
    fn from_i32s(v: int32x4_t) -> __m128i {
        // SAFETY: as above.
        __m128i(unsafe { vreinterpretq_u8_s32(v) })
    }
    #[inline(always)]
    fn lanes_f32(a: __m128) -> [f32; 4] {
        // SAFETY: `__m128` is exactly four f32 lanes.
        unsafe { std::mem::transmute::<float32x4_t, [f32; 4]>(a.0) }
    }
    #[inline(always)]
    fn from_lanes_f32(v: [f32; 4]) -> __m128 {
        // SAFETY: as above.
        __m128(unsafe { std::mem::transmute::<[f32; 4], float32x4_t>(v) })
    }
    #[inline(always)]
    fn lanes_u32(a: __m128i) -> [u32; 4] {
        // SAFETY: 128 bits as four u32 lanes.
        unsafe { std::mem::transmute::<uint8x16_t, [u32; 4]>(a.0) }
    }
    #[inline(always)]
    fn from_lanes_u32(v: [u32; 4]) -> __m128i {
        // SAFETY: as above.
        __m128i(unsafe { std::mem::transmute::<[u32; 4], uint8x16_t>(v) })
    }
    #[inline(always)]
    fn lanes_u16(a: __m128i) -> [u16; 8] {
        // SAFETY: 128 bits as eight u16 lanes.
        unsafe { std::mem::transmute::<uint8x16_t, [u16; 8]>(a.0) }
    }
    #[inline(always)]
    fn from_lanes_u16(v: [u16; 8]) -> __m128i {
        // SAFETY: as above.
        __m128i(unsafe { std::mem::transmute::<[u16; 8], uint8x16_t>(v) })
    }
    #[inline(always)]
    fn lanes_u8(a: __m128i) -> [u8; 16] {
        // SAFETY: 128 bits as sixteen bytes.
        unsafe { std::mem::transmute::<uint8x16_t, [u8; 16]>(a.0) }
    }
    #[inline(always)]
    fn from_lanes_u8(v: [u8; 16]) -> __m128i {
        // SAFETY: as above.
        __m128i(unsafe { std::mem::transmute::<[u8; 16], uint8x16_t>(v) })
    }

    // ---- f32 arithmetic, comparison, logic -------------------------------

    #[inline(always)]
    pub unsafe fn _mm_setzero_ps() -> __m128 {
        __m128(vdupq_n_f32(0.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_set1_ps(a: f32) -> __m128 {
        __m128(vdupq_n_f32(a))
    }
    #[inline(always)]
    pub unsafe fn _mm_add_ps(a: __m128, b: __m128) -> __m128 {
        __m128(vaddq_f32(f(a), f(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_sub_ps(a: __m128, b: __m128) -> __m128 {
        __m128(vsubq_f32(f(a), f(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_mul_ps(a: __m128, b: __m128) -> __m128 {
        __m128(vmulq_f32(f(a), f(b)))
    }
    /// SSE `maxps`: `a > b ? a : b` per lane, so `b` when either is NaN (NEON's
    /// `fmax` would propagate the NaN instead).
    #[inline(always)]
    pub unsafe fn _mm_max_ps(a: __m128, b: __m128) -> __m128 {
        __m128(vbslq_f32(vcgtq_f32(f(a), f(b)), f(a), f(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128 {
        from_u32s(vcgtq_f32(f(a), f(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128 {
        from_u32s(vcleq_f32(f(a), f(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_and_ps(a: __m128, b: __m128) -> __m128 {
        from_u32s(vandq_u32(u32s(a), u32s(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_or_ps(a: __m128, b: __m128) -> __m128 {
        from_u32s(vorrq_u32(u32s(a), u32s(b)))
    }
    /// `(!a) & b`.
    #[inline(always)]
    pub unsafe fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128 {
        from_u32s(vbicq_u32(u32s(b), u32s(a)))
    }
    /// Lane sign bits in bits 0..4.
    #[inline(always)]
    pub unsafe fn _mm_movemask_ps(a: __m128) -> i32 {
        let l = lanes_f32(a);
        (0..4).fold(0, |m, i| m | (((l[i].to_bits() >> 31) as i32) << i))
    }
    /// Lane 0 from `b`, lanes 1..4 from `a`.
    #[inline(always)]
    pub unsafe fn _mm_move_ss(a: __m128, b: __m128) -> __m128 {
        __m128(vsetq_lane_f32::<0>(vgetq_lane_f32::<0>(f(b)), f(a)))
    }
    /// Lanes 0,1 picked from `a` and lanes 2,3 from `b` by the four 2-bit fields
    /// of `MASK`, low field first.
    #[inline(always)]
    pub unsafe fn _mm_shuffle_ps<const MASK: i32>(a: __m128, b: __m128) -> __m128 {
        let (a, b) = (lanes_f32(a), lanes_f32(b));
        let m = MASK as usize;
        from_lanes_f32([a[m & 3], a[(m >> 2) & 3], b[(m >> 4) & 3], b[(m >> 6) & 3]])
    }

    // ---- f32 loads and stores --------------------------------------------

    #[inline(always)]
    pub unsafe fn _mm_load_ps(p: *const f32) -> __m128 {
        __m128(vld1q_f32(p))
    }
    #[inline(always)]
    pub unsafe fn _mm_loadu_ps(p: *const f32) -> __m128 {
        __m128(vld1q_f32(p))
    }
    #[inline(always)]
    pub unsafe fn _mm_store_ps(p: *mut f32, a: __m128) {
        vst1q_f32(p, f(a))
    }
    #[inline(always)]
    pub unsafe fn _mm_storeu_ps(p: *mut f32, a: __m128) {
        vst1q_f32(p, f(a))
    }
    #[inline(always)]
    pub unsafe fn _mm_store_ss(p: *mut f32, a: __m128) {
        *p = vgetq_lane_f32::<0>(f(a));
    }

    // ---- conversions and casts -------------------------------------------

    #[inline(always)]
    pub unsafe fn _mm_castps_si128(a: __m128) -> __m128i {
        __m128i(vreinterpretq_u8_f32(a.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_castsi128_ps(a: __m128i) -> __m128 {
        __m128(vreinterpretq_f32_u8(a.0))
    }
    /// Truncating conversion; NaN and out-of-range lanes become `i32::MIN`, the
    /// x86 "integer indefinite" value (NEON would saturate instead).
    #[inline(always)]
    pub unsafe fn _mm_cvttps_epi32(a: __m128) -> __m128i {
        let l = lanes_f32(a);
        let lane = |x: f32| {
            if x.is_nan() || x >= 2_147_483_648.0 || x < -2_147_483_648.0 {
                i32::MIN as u32
            } else {
                x as i32 as u32
            }
        };
        from_lanes_u32([lane(l[0]), lane(l[1]), lane(l[2]), lane(l[3])])
    }
    /// Round-to-nearest conversion, as `cvtdq2ps` under the default MXCSR.
    #[inline(always)]
    pub unsafe fn _mm_cvtepi32_ps(a: __m128i) -> __m128 {
        __m128(vcvtq_f32_s32(i32s(a)))
    }
    #[inline(always)]
    pub unsafe fn _mm_cvtsi128_si32(a: __m128i) -> i32 {
        vgetq_lane_s32::<0>(i32s(a))
    }

    // ---- integer construction, loads, stores -----------------------------

    #[inline(always)]
    pub unsafe fn _mm_setzero_si128() -> __m128i {
        __m128i(vdupq_n_u8(0))
    }
    #[inline(always)]
    pub unsafe fn _mm_set1_epi8(a: i8) -> __m128i {
        __m128i(vdupq_n_u8(a as u8))
    }
    #[inline(always)]
    pub unsafe fn _mm_set1_epi16(a: i16) -> __m128i {
        from_i16s(vdupq_n_s16(a))
    }
    #[inline(always)]
    pub unsafe fn _mm_set1_epi32(a: i32) -> __m128i {
        from_i32s(vdupq_n_s32(a))
    }
    #[inline(always)]
    pub unsafe fn _mm_loadu_si128(p: *const __m128i) -> __m128i {
        __m128i(vld1q_u8(p as *const u8))
    }
    #[inline(always)]
    pub unsafe fn _mm_storeu_si128(p: *mut __m128i, a: __m128i) {
        vst1q_u8(p as *mut u8, a.0)
    }

    // ---- integer arithmetic and comparison -------------------------------

    #[inline(always)]
    pub unsafe fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vorrq_u8(a.0, b.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vmaxq_u8(a.0, b.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vqaddq_u8(a.0, b.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vqsubq_u8(a.0, b.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vreinterpretq_u8_s8(vqsubq_s8(
            vreinterpretq_s8_u8(a.0),
            vreinterpretq_s8_u8(b.0),
        )))
    }
    #[inline(always)]
    pub unsafe fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vceqq_u8(a.0, b.0))
    }
    #[inline(always)]
    pub unsafe fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
        from_i16s(vmaxq_s16(i16s(a), i16s(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
        from_i16s(vqaddq_s16(i16s(a), i16s(b)))
    }
    #[inline(always)]
    pub unsafe fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i {
        __m128i(vreinterpretq_u8_u16(vcgtq_s16(i16s(a), i16s(b))))
    }
    #[inline(always)]
    pub unsafe fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
        from_i32s(vaddq_s32(i32s(a), i32s(b)))
    }
    /// Logical left shift of each dword; counts above 31 give zero, as on x86.
    /// Takes the count as an argument, matching the legacy call form the
    /// kernels use for this one intrinsic.
    #[inline(always)]
    pub unsafe fn _mm_slli_epi32(a: __m128i, imm8: i32) -> __m128i {
        if !(0..32).contains(&imm8) {
            return _mm_setzero_si128();
        }
        __m128i(vreinterpretq_u8_u32(vshlq_u32(
            vreinterpretq_u32_u8(a.0),
            vdupq_n_s32(imm8),
        )))
    }
    /// Logical right shift of each dword.
    #[inline(always)]
    pub unsafe fn _mm_srli_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
        if !(0..32).contains(&IMM8) {
            return _mm_setzero_si128();
        }
        __m128i(vreinterpretq_u8_u32(vshlq_u32(
            vreinterpretq_u32_u8(a.0),
            vdupq_n_s32(-IMM8),
        )))
    }
    /// Byte `i` of the result is the sign bit of byte `i`.
    #[inline(always)]
    pub unsafe fn _mm_movemask_epi8(a: __m128i) -> i32 {
        let b = lanes_u8(a);
        (0..16).fold(0, |m, i| m | (((b[i] >> 7) as i32) << i))
    }
    /// Unsigned 16-bit lane `IMM8`, zero-extended.
    #[inline(always)]
    pub unsafe fn _mm_extract_epi16<const IMM8: i32>(a: __m128i) -> i32 {
        lanes_u16(a)[(IMM8 & 7) as usize] as i32
    }

    // ---- whole-register byte shifts and lane shuffles --------------------

    /// Shift the 128 bits right by `IMM8` bytes, filling with zero.
    #[inline(always)]
    pub unsafe fn _mm_srli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
        let z = vdupq_n_u8(0);
        // The kernels shift by these amounts; each arm is one `ext`.
        match IMM8 {
            0 => a,
            1 => __m128i(vextq_u8::<1>(a.0, z)),
            2 => __m128i(vextq_u8::<2>(a.0, z)),
            4 => __m128i(vextq_u8::<4>(a.0, z)),
            8 => __m128i(vextq_u8::<8>(a.0, z)),
            14 => __m128i(vextq_u8::<14>(a.0, z)),
            _ => {
                let b = lanes_u8(a);
                let n = IMM8.clamp(0, 16) as usize;
                let mut out = [0u8; 16];
                out[..16 - n].copy_from_slice(&b[n..]);
                from_lanes_u8(out)
            }
        }
    }
    /// Shift the 128 bits left by `IMM8` bytes, filling with zero.
    #[inline(always)]
    pub unsafe fn _mm_slli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
        let z = vdupq_n_u8(0);
        match IMM8 {
            0 => a,
            1 => __m128i(vextq_u8::<15>(z, a.0)),
            2 => __m128i(vextq_u8::<14>(z, a.0)),
            4 => __m128i(vextq_u8::<12>(z, a.0)),
            8 => __m128i(vextq_u8::<8>(z, a.0)),
            _ => {
                let b = lanes_u8(a);
                let n = IMM8.clamp(0, 16) as usize;
                let mut out = [0u8; 16];
                out[n..].copy_from_slice(&b[..16 - n]);
                from_lanes_u8(out)
            }
        }
    }
    /// Dword `i` of the result is dword `(IMM8 >> 2i) & 3` of `a`.
    #[inline(always)]
    pub unsafe fn _mm_shuffle_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
        let l = lanes_u32(a);
        let m = IMM8 as usize;
        from_lanes_u32([l[m & 3], l[(m >> 2) & 3], l[(m >> 4) & 3], l[(m >> 6) & 3]])
    }
    /// Words 0..4 shuffled as `_mm_shuffle_epi32` does dwords; words 4..8 kept.
    #[inline(always)]
    pub unsafe fn _mm_shufflelo_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
        let l = lanes_u16(a);
        let m = IMM8 as usize;
        from_lanes_u16([
            l[m & 3],
            l[(m >> 2) & 3],
            l[(m >> 4) & 3],
            l[(m >> 6) & 3],
            l[4],
            l[5],
            l[6],
            l[7],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f4(v: __m128) -> [f32; 4] {
        let mut out = [0.0; 4];
        // SAFETY: storing four lanes into a four-element array.
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), v) };
        out
    }

    fn b16(v: __m128i) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: storing 16 bytes into a 16-byte array.
        unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, v) };
        out
    }

    fn load_f(a: [f32; 4]) -> __m128 {
        // SAFETY: loading four lanes from a four-element array.
        unsafe { _mm_loadu_ps(a.as_ptr()) }
    }

    fn load_b(a: [u8; 16]) -> __m128i {
        // SAFETY: loading 16 bytes from a 16-byte array.
        unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
    }

    const BYTES: [u8; 16] = [
        0, 1, 2, 3, 4, 5, 6, 7, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    ];

    /// These expectations are SSE's documented behavior; the x86 run checks
    /// them against the hardware and the aarch64 run against the NEON shim.
    #[test]
    fn edge_cases_follow_sse_semantics() {
        // SAFETY: SSE2 / NEON are baseline on the targets this module builds for.
        unsafe {
            let nan = f32::NAN;
            let m = f4(_mm_max_ps(
                load_f([nan, 1.0, 3.0, -0.0]),
                load_f([2.0, nan, 1.0, 0.0]),
            ));
            assert_eq!(m[0], 2.0);
            assert!(m[1].is_nan());
            assert_eq!(m[2], 3.0);
            assert_eq!(m[3].to_bits(), 0.0f32.to_bits());

            let t = b16(_mm_cvttps_epi32(load_f([1.9, -1.9, nan, 3.0e9])));
            let t: [i32; 4] = std::mem::transmute(t);
            assert_eq!(t, [1, -1, i32::MIN, i32::MIN]);

            let s = f4(_mm_shuffle_ps::<{ crate::simd::shuffle_mask(1, 0, 3, 2) }>(
                load_f([0.0, 1.0, 2.0, 3.0]),
                load_f([4.0, 5.0, 6.0, 7.0]),
            ));
            assert_eq!(s, [2.0, 3.0, 4.0, 5.0]);

            let a = load_b(BYTES);
            assert_eq!(b16(_mm_srli_si128::<4>(a))[..12], BYTES[4..]);
            assert_eq!(b16(_mm_srli_si128::<4>(a))[12..], [0; 4]);
            assert_eq!(b16(_mm_slli_si128::<2>(a))[..2], [0; 2]);
            assert_eq!(b16(_mm_slli_si128::<2>(a))[2..], BYTES[..14]);
            assert_eq!(_mm_movemask_epi8(a), 0xff00);
            assert_eq!(_mm_extract_epi16::<7>(a), 0xffee);
            let sat = b16(_mm_adds_epi16(_mm_set1_epi16(32000), _mm_set1_epi16(1000)));
            assert_eq!(&sat[..2], &32767i16.to_le_bytes());
            assert_eq!(
                b16(_mm_subs_epu8(_mm_set1_epi8(3), _mm_set1_epi8(5))),
                [0; 16]
            );
            assert_eq!(_mm_movemask_ps(load_f([-1.0, 1.0, -0.0, 0.0])), 0b0101);
            let lo = b16(_mm_shufflelo_epi16::<0x4E>(a));
            assert_eq!(lo[..8], [4, 5, 6, 7, 0, 1, 2, 3]);
            assert_eq!(lo[8..], BYTES[8..]);
            let k = b16(_mm_slli_epi32(_mm_set1_epi32(1), 23));
            assert_eq!(&k[..4], &(1u32 << 23).to_le_bytes());
            let r = b16(_mm_srli_epi32::<16>(_mm_set1_epi32(0x1234_5678)));
            assert_eq!(&r[..4], &0x1234u32.to_le_bytes());
        }
    }
}
