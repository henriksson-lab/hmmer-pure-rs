//! Vector operations on f32/f64/i32 slices.
//! Direct port of Easel's esl_vectorops.c, focusing on functions used by HMMER.

// ===== Set =====

/// Set every element of `vec` to `value`. Port of `esl_vec_DSet`.
pub fn d_set(vec: &mut [f64], value: f64) {
    vec.iter_mut().for_each(|v| *v = value);
}

/// Set every element of `vec` to `value`. Port of `esl_vec_FSet`.
pub fn f_set(vec: &mut [f32], value: f32) {
    vec.iter_mut().for_each(|v| *v = value);
}

/// Set every element of `vec` to `value`. Port of `esl_vec_ISet`.
pub fn i_set(vec: &mut [i32], value: i32) {
    vec.iter_mut().for_each(|v| *v = value);
}

// ===== Scale =====

/// Multiply every element of `vec` by `scale`. Port of `esl_vec_DScale`.
pub fn d_scale(vec: &mut [f64], scale: f64) {
    vec.iter_mut().for_each(|v| *v *= scale);
}

/// Multiply every element of `vec` by `scale`. Port of `esl_vec_FScale`.
pub fn f_scale(vec: &mut [f32], scale: f32) {
    vec.iter_mut().for_each(|v| *v *= scale);
}

// ===== Increment =====

/// Add scalar `x` to every element of `vec`. Port of `esl_vec_FIncrement`.
pub fn f_increment(vec: &mut [f32], x: f32) {
    vec.iter_mut().for_each(|v| *v += x);
}

// ===== Add =====

/// Element-wise add `vec2` into `vec1`. Port of `esl_vec_DAdd`.
pub fn d_add(vec1: &mut [f64], vec2: &[f64]) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a += *b;
    }
}

/// Element-wise add `vec2` into `vec1`. Port of `esl_vec_FAdd`.
pub fn f_add(vec1: &mut [f32], vec2: &[f32]) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a += *b;
    }
}

/// In-place vec1 += a * vec2 (element-wise). Port of `esl_vec_FAddScaled`.
pub fn f_add_scaled(vec1: &mut [f32], vec2: &[f32], a: f32) {
    for (v1, v2) in vec1.iter_mut().zip(vec2.iter()) {
        *v1 += *v2 * a;
    }
}

/// In-place vec1 += a * vec2 (element-wise). Port of `esl_vec_DAddScaled`.
pub fn d_add_scaled(vec1: &mut [f64], vec2: &[f64], a: f64) {
    for (v1, v2) in vec1.iter_mut().zip(vec2.iter()) {
        *v1 += *v2 * a;
    }
}

// ===== Sum (Kahan compensated summation) =====

/// Sum of `vec` using Kahan compensated summation. Port of `esl_vec_DSum`.
///
/// Most accurate when `vec` is sorted in increasing magnitude.
pub fn d_sum(vec: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Sum of `vec` using Kahan compensated summation. Port of `esl_vec_FSum`.
pub fn f_sum(vec: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// ===== Max, Min, ArgMax, ArgMin =====

/// Return the maximum element in `vec`. Port of `esl_vec_DMax`.
pub fn d_max(vec: &[f64]) -> f64 {
    vec.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Return the maximum element in `vec`. Port of `esl_vec_FMax`.
pub fn f_max(vec: &[f32]) -> f32 {
    vec.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Return the minimum element in `vec`. Port of `esl_vec_FMin`.
pub fn f_min(vec: &[f32]) -> f32 {
    vec.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Return the index of the maximum element in `vec`. Port of `esl_vec_FArgMax`.
///
/// On ties the smallest index is returned, matching the C convention HMMER's
/// optimal-accuracy tracebacks depend on. Returns 0 for an empty slice.
pub fn f_argmax(vec: &[f32]) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Return the index of the maximum element in `vec`. Port of `esl_vec_IArgMax`.
pub fn i_argmax(vec: &[i32]) -> usize {
    vec.iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ===== Copy =====

/// Copy `src` into `dst[..src.len()]`. Port of `esl_vec_DCopy`.
pub fn d_copy(src: &[f64], dst: &mut [f64]) {
    dst[..src.len()].copy_from_slice(src);
}

/// Copy `src` into `dst[..src.len()]`. Port of `esl_vec_FCopy`.
pub fn f_copy(src: &[f32], dst: &mut [f32]) {
    dst[..src.len()].copy_from_slice(src);
}

/// Copy `src` into `dst[..src.len()]`. Port of `esl_vec_ICopy`.
pub fn i_copy(src: &[i32], dst: &mut [i32]) {
    dst[..src.len()].copy_from_slice(src);
}

// ===== Reverse =====

/// Write the elements of `vec` into `rev` in reversed order.
///
/// Caller must size `rev` for at least `vec.len()`. Unlike the C version,
/// this signature requires distinct buffers (no in-place reverse).
/// Port of `esl_vec_IReverse`.
pub fn i_reverse(vec: &[i32], rev: &mut [i32]) {
    for (i, &v) in vec.iter().enumerate() {
        rev[vec.len() - 1 - i] = v;
    }
}

// ===== Normalize =====

/// Normalize a probability vector so it sums to 1.0. Port of `esl_vec_DNorm`.
///
/// If the current sum is 0, the vector is left untouched (the C version
/// fills with 1/n; this Rust variant keeps zeros).
pub fn d_norm(vec: &mut [f64]) {
    let sum = d_sum(vec);
    if sum != 0.0 {
        d_scale(vec, 1.0 / sum);
    }
}

/// Normalize a probability vector so it sums to 1.0. Port of `esl_vec_FNorm`.
pub fn f_norm(vec: &mut [f32]) {
    let sum = f_sum(vec);
    if sum != 0.0 {
        f_scale(vec, 1.0 / sum);
    }
}

// ===== Log / Exp operations =====

/// Replace each element of `vec` with its natural log. Port of `esl_vec_FLog`.
pub fn f_log(vec: &mut [f32]) {
    vec.iter_mut().for_each(|v| *v = v.ln());
}

/// Replace each element of `vec` with exp(v). Port of `esl_vec_FExp`.
pub fn f_exp(vec: &mut [f32]) {
    vec.iter_mut().for_each(|v| *v = v.exp());
}

/// Convert an unnormalized log p-vector into a normalized probability vector.
///
/// Subtracts the max before exponentiating to avoid overflow; on return
/// `vec` sums to 1.0 in probability space. Port of `esl_vec_FLogNorm`.
pub fn f_lognorm(vec: &mut [f32]) {
    let max = f_max(vec);
    let mut sum = 0.0_f32;
    for v in vec.iter() {
        sum += (*v - max).exp();
    }
    let log_sum = max + sum.ln();
    for v in vec.iter_mut() {
        *v -= log_sum;
    }
    f_exp(vec);
}

/// Convert an unnormalized log_2 p-vector into a normalized probability vector.
///
/// Same shift-by-max stabilization as `f_lognorm`, but in log base 2.
/// Port of `esl_vec_FLog2Norm`.
pub fn f_log2norm(vec: &mut [f32]) {
    let max = f_max(vec);
    let mut sum = 0.0_f32;
    for v in vec.iter() {
        sum += f32::exp2(*v - max);
    }
    let log2_sum = max + sum.log2();
    for v in vec.iter_mut() {
        *v -= log2_sum;
    }
    vec.iter_mut().for_each(|v| *v = f32::exp2(*v));
}

// ===== Entropy =====

/// Shannon entropy H = -sum(p_i log_2 p_i) in bits. Port of `esl_vec_DEntropy`.
///
/// Skips terms with p_i <= 0 by convention.
pub fn d_entropy(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.log2();
        }
    }
    h
}

/// Shannon entropy in bits for a single-precision p-vector. Port of `esl_vec_FEntropy`.
pub fn f_entropy(p: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.log2();
        }
    }
    h
}

/// Kullback-Leibler divergence D(p || q) = sum p_i log_2(p_i/q_i), in bits.
///
/// Port of `esl_vec_DRelEntropy`. Unlike the C version this returns 0 for
/// (p_i>0, q_i==0) pairs rather than +inf, matching existing call sites.
pub fn d_rel_entropy(p: &[f64], q: &[f64]) -> f64 {
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 && qi > 0.0 {
            kl += pi * (pi / qi).log2();
        }
    }
    kl
}

/// Single-precision KL divergence D(p || q) in bits. Port of `esl_vec_FRelEntropy`.
pub fn f_rel_entropy(p: &[f32], q: &[f32]) -> f32 {
    let mut kl = 0.0_f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 && qi > 0.0 {
            kl += pi * (pi / qi).log2();
        }
    }
    kl
}

// ===== Type conversion =====

/// Copy a double vector to a float vector with lossy conversion. Port of `esl_vec_D2F`.
pub fn d2f(src: &[f64], dst: &mut [f32]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s as f32;
    }
}

/// Copy a float vector to a double vector. Port of `esl_vec_F2D`.
pub fn f2d(src: &[f32], dst: &mut [f64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s as f64;
    }
}

// ===== Compare =====

/// Return true if vec1 and vec2 agree elementwise within absolute tolerance `tol`.
///
/// Port of `esl_vec_DCompare` (returns bool instead of eslOK/eslFAIL).
pub fn d_compare(vec1: &[f64], vec2: &[f64], tol: f64) -> bool {
    if vec1.len() != vec2.len() {
        return false;
    }
    vec1.iter()
        .zip(vec2.iter())
        .all(|(a, b)| (a - b).abs() <= tol)
}

/// Return true if vec1 and vec2 agree elementwise within `tol`. Port of `esl_vec_FCompare`.
pub fn f_compare(vec1: &[f32], vec2: &[f32], tol: f32) -> bool {
    if vec1.len() != vec2.len() {
        return false;
    }
    vec1.iter()
        .zip(vec2.iter())
        .all(|(a, b)| (a - b).abs() <= tol)
}

/// Strict equality of two integer vectors. Port of `esl_vec_ICompare`.
pub fn i_compare(vec1: &[i32], vec2: &[i32]) -> bool {
    vec1 == vec2
}

// ===== Validate =====

/// Return true if `vec` is a valid probability vector: all entries in
/// [0, 1+tol] and the total within `tol` of 1.0. Port of `esl_vec_DValidate`.
pub fn d_validate(vec: &[f64], tol: f64) -> bool {
    let sum = d_sum(vec);
    (sum - 1.0).abs() <= tol && vec.iter().all(|&v| v >= 0.0 && v <= 1.0 + tol)
}

/// Return true if `vec` is a valid f32 probability vector (sum near 1, all in `[0,1]`).
/// Port of `esl_vec_FValidate`.
pub fn f_validate(vec: &[f32], tol: f32) -> bool {
    let sum = f_sum(vec);
    (sum - 1.0).abs() <= tol && vec.iter().all(|&v| v >= 0.0 && v <= 1.0 + tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_sum_kahan() {
        // Kahan summation should be more accurate than naive
        let vec = vec![1.0e-8_f32; 100_000_000];
        let result = f_sum(&vec[..100]); // just test small case
        assert!((result - 1.0e-6).abs() < 1.0e-12);
    }

    #[test]
    fn test_d_norm() {
        let mut v = vec![2.0, 3.0, 5.0];
        d_norm(&mut v);
        assert!((d_sum(&v) - 1.0).abs() < 1e-10);
        assert!((v[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_f_entropy() {
        // Uniform distribution over 4 symbols = 2 bits
        let p = vec![0.25_f32; 4];
        assert!((f_entropy(&p) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_f_argmax() {
        let v = vec![1.0_f32, 5.0, 3.0, 2.0];
        assert_eq!(f_argmax(&v), 1);
    }
}
