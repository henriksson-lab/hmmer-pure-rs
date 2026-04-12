# --seed Reproducibility

## Status

Fixed:
- Forward tau formula: restored `+ log(tailp)/lambda` correction term
- FLAMBDA: all three use HMM-derived lambda (not fitted Gumbel lambda)
- Lambda computation: H now computed in bits (log2) matching C's `p7_MeanMatchRelativeEntropy()` — lambda matches C exactly (0.72049)
- Profile log-odds: use f64 intermediate precision matching C's `log((double)...)`

Remaining difference: MSV mu, Viterbi mu, and Forward tau may differ slightly from C's values when recalibrating the same HMM due to:

1. Our Mersenne Twister RNG matches C's (verified) → identical random sequences
2. SIMD byte/word scores may differ by ±1 at float rounding boundaries
3. Gumbel fitting amplifies small score differences into mu/tau shifts

## Impact

- For HMMs read from C-generated files (Pfam, etc.): **no issue** — we use the pre-calibrated evparams
- For HMMs built by our hmmbuild: E-values differ by ~10-30% from C's
- Same hits are found at standard E-value thresholds (10.0 default)

## Fix approach

To close the remaining gap, ensure `biased_byteify()` and `wordify()` produce bit-identical values to C for every emission/transition score. The current implementation matches for most values but may differ at float rounding boundaries where `roundf(scale * score)` gives `N` in C and `N±1` in Rust.
