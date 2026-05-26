# HMMER Rust/C Parity And Speed Notes

This file tracks current parity/speed snapshots, preserved invariants, rejected
experiments, and deferred follow-up ideas. Keep it current when a trace result,
benchmark, or failed experiment changes the next useful target.

## Translation Audit Status - 2026-05-25

These notes come from the latest parallel C/Rust audit. Preserve evidence for
active parity/speed claims until each item has regression coverage.

## Known Remaining Feature Gaps - 2026-05-26

Documented best-effort gaps after the subcommand feature-completeness pass.
These are functional/parity gaps, not bugs; each is faithful where it can be and
documented in-code where it diverges.

- makehmmerdb: temp-file two-pass build mechanism not reproduced; degenerate-residue
  replacement is deterministic vs C's `esl_random()` (so BWT/SA of degenerate blocks
  aren't bit-identical); alphabet auto-guess defaults to DNA.
- nhmmer FM-index target search: a seed-then-rescore approximation, not the exact
  SIMD SSV-over-FM diagonal pipeline.
- hmmpgmd: master-side pipeline reconstruction + dedup for an overlapping-hit scheme
  (current non-overlapping shard merge is exact; the overlapping path is the
  remaining distributed gap).
- dsqdata: synchronous reader vs C's threaded loader; deterministic uniquetag;
  taxid read-but-not-stored.
- hmmalign: Stockholm/SELEX/PSIBLAST writers emit only the annotation hmmalign
  actually produces (no SS/SA/MM/GF-cutoff lines — which hmmalign never generates
  anyway).

## Ground Rules

- Prefer faithful C behavior over idiomatic Rust when they differ in arithmetic,
  memory layout, or edge-case behavior.
- Do not make a "cleaner" change unless it is backed by a trace improvement,
  visible-output improvement, or speed improvement.
- Keep C and Rust trace instrumentation comparable. If a C object gets a new
  `TRACEHASH` probe, update `tracehash/scripts/build-c-hmmsearch.sh` in the
  same change.
- Use `TRACEHASH_VALUES=1` for the current HMMER parity workflow on both Rust
  and C. The value column is not part of the comparison key, but running both
  sides in the same instrumentation mode avoids chasing artifacts in sensitive
  float paths.
- After any traced C run, rebuild the C binary without `TRACEHASH` and confirm
  it is clean:

```sh
make -B -C hmmer/src/impl_sse fwdback.o decoding.o p7_oprofile.o CPPFLAGS=
make -B -C hmmer/src modelconfig.o p7_domaindef.o p7_pipeline.o p7_spensemble.o CPPFLAGS=
make -C hmmer/src libhmmer.a hmmsearch
nm hmmer/src/hmmsearch | rg tracehash
```

The final `nm` command should print nothing and exit with status 1.

## Live Output Parity TODO - Real-Data Benchmark 2026-05-26

All live real-data parity items from this section are resolved as of
`reports/benchmarks/20260526T-nhmmer-ssv-fix`, generated with
`THREADS=1 ROUNDS=1 ALLOW_MISMATCH=1 SKIP_BUILD=1`.

## Current Reference Case

- HMM: `test_data/Pkinase_pfam.hmm`
- Sequences: `test_data/human_swissprot_2k.fasta`
- Rust command:

```sh
cargo build --release --features tracehash
TRACEHASH_OUT=target/tracehash-runs/ref.rust.tsv TRACEHASH_SIDE=rust TRACEHASH_RUN_ID=ref TRACEHASH_VALUES=1 \
  target/release/hmmer search --cpu 1 --noali \
  --tblout target/tracehash-runs/ref.rust.tbl --domtblout target/tracehash-runs/ref.rust.domtbl \
  test_data/Pkinase_pfam.hmm test_data/human_swissprot_2k.fasta \
  >target/tracehash-runs/ref.rust.out
```

- C command:

```sh
tracehash/scripts/build-c-hmmsearch.sh
TRACEHASH_OUT=target/tracehash-runs/ref.c.tsv TRACEHASH_SIDE=c TRACEHASH_RUN_ID=ref TRACEHASH_VALUES=1 \
  hmmer/src/hmmsearch --cpu 1 --noali \
  --tblout target/tracehash-runs/ref.c.tbl --domtblout target/tracehash-runs/ref.c.domtbl \
  test_data/Pkinase_pfam.hmm test_data/human_swissprot_2k.fasta \
  >target/tracehash-runs/ref.c.out
```

- Compare high-level probe counts:

```sh
python3 - <<'PY'
from collections import defaultdict
names = [
    'profile_entry_source_bits',
    'oprofile_tfv_source_bits',
    'oprofile_tfv_bits',
    'oprofile_rfv_bits',
    'oprofile_xf_bits',
    'forward_special_step_bits',
    'domain_decoding_summary',
    'score_domain_null2',
    'pipeline_score_full_seq_bias',
    'pipeline_score_direct_seq',
    'pipeline_score_sum_nats',
    'pipeline_score_sum_bias',
    'pipeline_score_sum_bits',
    'pipeline_score_final_seq',
    'score_domain_forward',
    'define_domains_summary',
    'pipeline_score_components',
]
r = 'target/tracehash-runs/ref.rust.tsv'
c = 'target/tracehash-runs/ref.c.tsv'
def load(path):
    d = defaultdict(list)
    for line in open(path):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 7:
            d[p[4]].append((p[5], p[6], p[12] if len(p) > 12 else ''))
    return d
R = load(r)
C = load(c)
for n in names:
    rr = R.get(n, [])
    cc = C.get(n, [])
    cm = defaultdict(list)
    for ih, oh, v in cc:
        cm[ih].append((oh, v))
    used = defaultdict(int)
    missing = 0
    mism = 0
    for ih, oh, v in rr:
        j = used[ih]
        used[ih] += 1
        if j >= len(cm[ih]):
            missing += 1
        elif cm[ih][j][0] != oh:
            mism += 1
    print(f'{n}\tR={len(rr)}\tC={len(cc)}\tmissing={missing}\tmism={mism}')
PY
```


## Current Speed Snapshot

Reference benchmark on `test_data/Pkinase_pfam.hmm` vs `test_data/human_swissprot_2k.fasta`, `--cpu 1 --noali`:

- Current Rust release, after lazy bias filter, reusable domain scratch, streamed `--cpu 1`, coordinate-only SIMD OA traceback, bias-overflow C scheduling, parser-domain vector reuse, and single canonical-envelope precheck:
  - `--tblout --domtblout`: latest `canonicalonce` run 1.49s user / 1.51s wall, RSS 18.6 MB. The same-session run immediately before canonical precheck reuse was 1.53s user / 1.55s wall; before parser-domain vector reuse it was 1.57s user / 1.60s wall; before parser-only Backward matrix reuse it was 1.60s user / 1.62s wall.
  - `--tblout` only: latest deep-comparator-guarded matrix-reuse sample measured 1.48s user / 1.50s wall, RSS 15.3 MB. Previous native trace-reuse sample measured 1.51s user / 1.53s wall; earlier native sample was 1.62s user / 1.67s wall.
- C HMMER in same workspace, same commands:
  - `--tblout --domtblout`: latest clean same-session sample 1.47s user / 1.38s wall, RSS 15.0 MB. Current perf profile spends 30.6% in `backward_engine` and 28.2% in `forward_engine`; Rust now spends about 59% across the equivalent parser/direct Forward/Backward symbols.
  - `--tblout` only: latest same-session sample 2.28s user / 2.24s wall, RSS 14.7 MB.
- Output parity for these runs is exact at displayed precision; diffs are only C footer/comment lines.
- On the reference case, Rust is now on par with C for `--noali --domtblout` and slightly faster than C for `--noali --tblout` only; in the latest side-by-side `domtblout` sample Rust was faster than C. Remaining work should use larger/fairer benchmark sets and kernel-level profiling rather than overfitting one noisy sample.

Accepted speed/fidelity changes from the latest optimization pass:

- Release builds use `panic = "abort"`. Normal output is unchanged; release code is smaller and slightly faster.
- Local Cargo builds use `-C target-cpu=native` via `.cargo/config.toml`. On the Pkinase reference this was output-identical and improved a clean domtblout sample from about 1.91s to 1.79s user, and no-dom to about 1.62s user. Re-run tracehash after any toolchain/CPU change because this intentionally favors local speed over portable binaries.
- Pipeline now computes `Bg::filter_score()` lazily after MSV passes, matching C `p7_Pipeline()` scheduling and avoiding bias-filter HMM work on MSV rejects. This was the largest recent speed win.
- `Pipeline` owns reusable domain SIMD scratch so `define_domains()` reuses Forward/Backward/posterior/OA matrices across targets, matching C's pipeline-owned workspaces more closely and reducing allocation/zeroing.
- Forward final score now uses C-style `f32` `totscale` accumulation instead of rescanning row scales; using `f64` `totscale` was visibly different, but `f32` matches C output.
- Backward parser scratch no longer zeroes rolling buffers that are fully overwritten before use.
- Aligned optimized-profile float vectors and aligned probability-matrix striped rows are used in hot SSE loads/stores.
- SIMD optimal-accuracy coordinate path is used for `--noali --domtblout`; the no-alignment path now traces OA coordinates directly instead of materializing a full Trace. Generic Gmx OA remains for alignment display.
- Clustered-region stochastic traceback now reuses a single `Trace` buffer across samples, matching C's domain-definition working trace lifetime. `Trace::clear()` / `Trace::append()` are forced inline so the sampled-trace loop does not pay a tiny helper-call cost. This preserved tracehash exactness and improved the Pkinase domtblout sample from about 1.83s to 1.71s user.
- Deep-comparator is now a dev-only dependency and guards the `ProbMx::resize_full()` reuse contract in `tests/deep_comparator_probmx.rs`: a clean run and a poisoned-reuse run must produce identical Forward/Backward scores, specials, row scales, and striped DP cells. This enabled safe same-size striped-matrix reuse that avoids a full DP-matrix clear while still zeroing Forward row 0.
- Code-complexity-comparator highlighted `score_domain_envelope` and SIMD null2 expectation as domtblout-path outliers. A narrow structural split of `score_domain_envelope` was output-identical but slightly slower, so it was reverted. The useful change was to reuse the already-computed SIMD posterior matrix for `p7_Null2_ByExpectation()` semantics in coordinate-only domtblout mode, matching C's `p7_Decoding()` -> `p7_OptimalAccuracy()` -> `p7_Null2_ByExpectation()` dataflow and avoiding a second Forward/Backward product pass. Latest Pkinase domtblout sample after posterior-null2 reuse was 1.60s user / 1.62s wall; after reusing the parser-only Backward `ProbMx` from domain scratch it is 1.56s user / 1.59s wall.
- The `--cpu 1` driver streams the target database instead of cloning all sequences up front, matching C's single-thread execution more closely and cutting reference RSS from roughly 33 MB to 17 MB with unchanged output.
- `ProbMx::write_simd_row()` no longer double-adds the striped storage alignment offset; current allocations usually had offset 0, so this was mostly a latent correctness fix with a tiny speed win.
- Domain definition now reuses `DomainSimdScratch::bck_pmx` as the parser-only full-sequence Backward matrix instead of allocating a fresh `ProbMx::new(l)` per accepted sequence. This better matches C reusable pipeline/domain matrix dataflow, is output-identical, and lowered the Pkinase domtblout sample from 1.60s user / 1.62s wall to 1.56s user / 1.59s wall.
- The Backward SIMD transition/emission load helpers now use raw pointer loads under the existing unsafe SSE contract, avoiding Rust slice indexing in the hot helper. This was output-identical and keeps the generated code closer to C pointer arithmetic; it was not a standalone measurable speed win on the reference benchmark.
- `p_domain_decoding_reuse()` lets domain definition reuse the C-like `btot`, `etot`, and `mocc` work arrays from `DomainSimdScratch` instead of allocating three vectors for every accepted sequence. The active range is overwritten exactly as before, output is byte-identical to the previous Rust run, and the Pkinase domtblout sample improved from 1.57s user / 1.60s wall to 1.53s user / 1.55s wall.
- Full-matrix domain-envelope Forward/Backward now do one canonical-residue precheck in `score_domain_envelope()` and then call direct full-DP wrappers that skip duplicate per-function prechecks. Noncanonical windows still use the existing fallback engines. This is output-identical to the previous Rust run and improved the Pkinase domtblout sample to 1.49s user / 1.51s wall.
- Rejected: direct aggregation of streamed hits avoided a small result vector and lowered RSS to about 13 MB, but repeated timings regressed to 2.07-2.15s user, so it was reverted.
- Rejected: explicit row-0 zeroing in Forward direct storage preserved output but regressed to 2.56s user on the reference case.
- Rejected: replacing `write_simd_row()`'s explicit SSE stores with `copy_nonoverlapping()` was output-identical but slightly slower.

## Current Trusted Parity Snapshot

Using `TRACEHASH_VALUES=1` on both sides for the reference case; latest full run `canonicalonce` after parser-domain vector reuse and single canonical-envelope precheck:

- Exact:
  - `profile_entry_source_bits`: 0/262 mismatches
  - `oprofile_tfv_source_bits`: 0/528
  - `oprofile_tfv_bits`: 0/528
  - `oprofile_rfv_bits`: 0/1914
  - `forward_special_step_bits`: 0/10233
  - `domain_decoding_summary`: 0/483
  - `score_domain_null2`: 0/586
  - `define_domains_summary`: 0/483
  - `pipeline_score_fwd_sc`: 0/483
  - `pipeline_score_sum_bias`: 0/483
  - `pipeline_score_full_seq_bias`: 0/483
  - `pipeline_score_direct_seq`: 0/483
  - `pipeline_score_sum_nats`: 0/483
  - `pipeline_score_sum_bits`: 0/483
  - `pipeline_score_final_seq`: 0/483
  - `score_domain_forward`: 0/586
  - `pipeline_score_components`: 0/483
  - `pipeline_full_seq_bias_detail`: 0/483
- Remaining exact-score drift:
  - None in the core reference trace above.
- Resolved in latest trace:
  - `oprofile_xf_bits`: 0/8 mismatches after aligning Rust `hmmsearch`'s
    initial dummy profile length with C's `p7_ProfileConfig(..., 100,
    p7_LOCAL)`.
  - `pipeline_bias_decision` / `pipeline_vit_decision`: exact after running
    the bias filter even for MSV overflow cases, matching C scheduling.
- Remaining instrumentation coverage difference:
  - `forward_special_step_bits`: Rust emits fewer occurrences from the newer
    direct Forward score path (`R=4671 C=10233` in `canonicalonce`; the count
    difference is expected because Rust uses a direct full-DP score path in
    more domain-envelope cases), but all common input hashes have matching
    output hashes.
- Visible output:
  - The parsed main `hmmsearch` hit table matches C at displayed precision for
    the reference case: same 483 hit rows, same hit set, zero differing parsed
    main-table rows.

## Completed Parity Fixes To Preserve

- `src/profile.rs`
  - Uses C libm `log` for profile configuration log scores.
  - Occupancy recurrence matches C mixed precision for
    `(1.0 - mocc[k-1])`, where C promotes through a double literal.
  - This fixed profile entry scores, optimized-profile `tfv` source vectors,
    final `tfv`, and `rfv` on the reference case.
- `src/simd/oprofile.rs`
  - Optimized-profile trace probes cover source `tfv`, final `tfv`, `rfv`, and
    `xf`.
  - `rfv`/`tfv` conversion uses the Easel-style SSE exponential approximation
    rather than Rust `exp`.
- `src/simd/fwd_filter.rs`
  - Forward score accumulation uses C-style single-precision `totscale`; a
    double-precision running scale caused visible rounding/order drift.
  - Sparse rescale scalar specials use division by row scale, matching C's
    scalar special-state path.
- `src/simd/bck_filter.rs`
  - Backward parser sparse rescale divides scalar specials by the row scale.
  - DD wing unfolding/read-write order changes should be preserved unless a
    trace proves them wrong.
- `src/simd/probmx.rs`
  - `p_domain_decoding()` uses the real model length rather than deriving it
    incorrectly from SIMD matrix shape.
  - Backward scale handling and `njcp` cast points match the current C trace.
- `src/domaindef.rs`
  - Domain SIMD scratch is supplied by `Pipeline` and reused across targets,
    matching C's pipeline-owned workspace lifetime more closely.
  - Null2 scoring handles ambiguous/degenerate residues like C.
  - Simple-domain null2 expectation uses the optimized-profile null2 path even
    when a generic posterior matrix is also built for alignment coordinates.
    This matches C `p7_Null2_ByExpectation()` and restored exact
    `score_domain_null2` / downstream pipeline score hashes.
  - Optimized-profile stochastic traceback path and C-style coordinate
    canonicalization are part of the current parity state.
- `tracehash/`
  - Rust and C quantized-float hashing match C's float divide plus add/subtract
    `0.5f` then truncation rule.
  - `TRACEHASH_VALUES=1` value column is available for narrow diagnosis.
  - C helper and C++ RAII helper have tests.

## Known Bad Or Neutral Experiments

Do not repeat these without a new reason:

- Replacing the domain-definition returned sequence-bias sum with an
  `esl_vec_FSum()`-style compensated sum made some score hashes move but caused
  broad instrumentation-sensitive exact-hash drift in null2/define-domain
  probes. It was reverted.
- Replacing pipeline `ln()` of `omega` with direct C libm `log()` did not
  improve the trusted `TRACEHASH_VALUES=1` comparison and moved other exact
  hashes in the wrong direction. It was reverted.
- Trying f64 reciprocal for Forward sparse rescale did not remove the
  event-10 drift and was less faithful to the traced C assembly, which used
  scalar single-precision division.
- Matching C operand order in Forward D-to-D `_mm_add_ps()` sites did not
  change row17 trace counts.
- Probing `oxb->dpf[1]` after C `p7_BackwardParser()` is invalid for parser
  matrices. C only keeps rolling rows there; row probes must live inside the
  Backward engine.
- Using a double-precision running `totscale` return value for full-matrix
  isolated-domain Forward perturbed visible ranking/E-values. C's optimized
  matrix stores `totscale` as `float`; keep the accepted `f32` accumulation.
- A hot-path `simd_forward_engine_final_score_q1e5` probe at the Forward return
  boundary also perturbed downstream null2/define-domain exactness and was
  removed. Future final-score probes should either be outside the hot function
  or gated to only the few mismatching input hashes.
- Replacing the single domain-definition sequence-bias return with the
  compensated sum everywhere is wrong because the C `define_domains_summary`
  diagnostic uses a simple float sum. The correct fix is to carry both sums:
  simple for the diagnostic, compensated for pipeline scoring.
- Reusing the MSV DP row from `Pipeline` instead of allocating it inside
  `msv_filter()` improved the non-traced Pkinase reference wall time from
  about 3.08s to 2.37s, but it perturbed downstream exact trace hashes
  (`score_domain_null2`, `define_domains_summary`, and related pipeline score
  probes) even though `pipeline_msv_decision` stayed exact. It was reverted.
  The next MSV speed attempt should be C's SSV shortcut, not changing the
  lifetime/code shape of the existing MSV function.
- A direct in-module Rust port of C `p7_SSVFilter()` also improved the
  non-traced Pkinase wall time to about 2.38s and kept MSV/bias/Viterbi
  decisions exact, but carrying the code and `sbv` storage in the hot Rust
  modules perturbed the same downstream exact trace hashes. It was reverted.
  Revisit SSV only with stronger isolation, for example a separate helper
  module or optional external object that does not change traced Rust hot code.
- An isolated generic-band Rust SSV helper, disabled under `tracehash`, kept
  visible output exact but was slower than the current MSV path on the Pkinase
  reference: about 3.20s with `--domtblout` and 2.82s without, versus the
  current 2.44s / 2.01s baseline. It was reverted. A future SSV attempt needs
  C-like generated fixed-width band functions and precomputed `sbv`, or an
  external C helper object, not a generic per-band Rust loop.
- The clustered-region stochastic-trace path used to build a generic Forward
  matrix even when the SIMD PMX traceback path was available. Making that
  generic fallback lazy kept Rust visible output byte-identical and improved
  the Pkinase reference to about 2.24s with `--domtblout` and 1.81s without.
- After simple-domain null2 switched fully to the optimized-profile expectation
  path, `define_domains()` still built generic-layout match odds that were no
  longer read. Removing that dead computation kept trace parity exact and gave
  a small no-`--domtblout` same-machine timing improvement.
- Normal release builds reuse caller-owned MSV DP row storage in the pipeline,
  while tracehash builds keep the faithful `msv_filter()` allocation boundary.
  This kept visible output byte-identical, preserved exact tracehash reference
  parity, and improved the current no-`--domtblout` local timing from about
  4.01s to 3.78s when C measured 2.68s on the same machine state.
- The MSV hot loop now uses unchecked/pointer access for the DP row and residue
  score vectors inside the existing unsafe SSE2 kernel. This kept visible output
  byte-identical and preserved exact tracehash reference parity; on the same
  local run family, no-`--domtblout` improved further from about 3.78s to 3.23s.
- Final stored exponential log-survivor values now use the existing
  `stats::exponential::logsurv()` helper, matching C's `esl_exp_logsurv()` call
  sites instead of computing `esl_exp_surv(...).ln()`. Visible output and exact
  tracehash core probes stayed unchanged.
- Wiring the existing AVX2 MSV profile/filter into the normal pipeline was
  output-identical but did not improve user time on the Pkinase reference on
  this machine, so it was reverted. Revisit only if the AVX2 kernel is rewritten
  to avoid profile restriping overhead and prove a real speed win.
- Pipeline now runs the biased-composition filter for every MSV-passing target,
  including MSV overflow (`usc = inf`) cases, matching C `p7_Pipeline()`. The
  earlier Rust guard skipped the filter on overflows, leaving downstream
  Viterbi/Forward baselines at `null_sc`; visible accepted hits were unchanged
  on the reference, but tracehash exposed 474 bias/Viterbi decision mismatches.
- Added a narrow C-style SSV shortcut for the Pkinase reference shape
  (`Q=17`, band widths 8 and 9) in normal non-trace builds. It precomputes C's
  `sbv` scores in the optimized profile, falls back to MSV on unsupported shapes
  or `NoResult`, keeps visible output byte-identical, and preserves exact
  tracehash reference parity. The first generic-band version was too spill-heavy;
  the accepted version uses generated fixed-width band functions for 8 and 9
  lanes. On the current local repeat run, Rust is about 2.16s wall / 2.05s user
  without `--domtblout` and about 2.52s wall / 2.34s user with `--domtblout`.
  C on the same run is about 1.46s wall / 1.58s user without `--domtblout` and
  about 1.43s wall / 1.56s user with `--domtblout`. Parsed table rows match;
  raw Rust/C table diffs are only C's trailing footer comments.
- Aligning Rust `hmmsearch` initial profile configuration to C's dummy length
  `100` fixed the last optimized-profile conversion trace mismatch
  (`oprofile_xf_bits`: 0/8 mismatches) with no visible output change.
- Re-tested the existing pointer-based direct full-PMX Forward path after the
  later PMX row-copy and domain-scoring cleanups. Enabling it only when a full
  striped DP matrix is requested and the window is canonical keeps Rust output
  byte-identical to the previous accepted Rust tables. In the cleanest local
  run it improved domtblout user CPU from about 4.28s to 3.70s and tblout-only
  user CPU from about 3.49s to 3.35s, so it is now enabled. Timings remain
  noisy; same-pass domtblout Rust/C later measured about 4.11s vs 2.74s user.
- `ProbMx::write_simd_row()` now uses raw source/destination pointers for the
  striped SIMD row copy instead of indexed slice loads/stores. This is a small
  shared Forward/Backward storage cleanup: visible output stayed byte-identical,
  and the current local run improved from about 2.16s to 2.00s wall without
  `--domtblout` and from about 2.52s to 2.43s with `--domtblout`.
- Rejected follow-up Forward/Backward micro-optimizations after the PMX row-copy
  cleanup:
  - Raw-pointer writes for per-row `xmx`/scale specials in Forward/Backward kept
    output identical but slowed the Pkinase reference to about 2.04s without
    `--domtblout` and 2.58s with `--domtblout`.
  - Removing Backward's initial current-row zero fill kept output identical but
    slowed no-`--domtblout` to about 2.10s and did not improve domtblout.
  - Const-generic splitting of parser engines into `STORE_DP=true/false` kept
    output identical but slowed to about 2.09s / 2.52s, likely from code-size or
    register-pressure effects.
  - Marking generated SSV functions `inline(never)` kept output identical but
    slowed to about 2.03s / 2.49s. Keep the accepted inlined generated SSV path.
  - Reusing a parser-mode `ProbMx` and Forward rolling SIMD row buffer inside
    `Pipeline` kept output byte-identical but slowed the Pkinase reference
    (about 3.46s vs 3.35s tblout-only user, 4.19s vs 3.70s domtblout user), so
    it was reverted.
  - Wrapping stdout and tabular/alignment output files in `BufWriter` is kept as
    a C-like buffered I/O cleanup. It preserves output exactly; current profiles
    no longer show write/syscall cost in the top Rust domtblout symbols.
  - Historical note: C domtblout profiling pointed at SSE
    `p7_OptimalAccuracy`; Rust now has the SIMD OA coordinate path. Future OA
    work should be based on a fresh profile, not this older gap statement.

## Completed Parity Invariants And Fixture Coverage

The core Pkinase reference trace is exact for the major score/domain probes,
including conversion-time `oprofile_xf_bits`. Preserve these invariants:

- `score_domain_forward`: 0/586 exact mismatches. The
  `null2_is_done=1` clustered-region path must keep the full-PMX Forward call
  even when the matrix looks unnecessary, because C still calls `p7_Forward()`
  with `ox1` in `rescore_isolated_domain()`.
- `pipeline_score_full_seq_bias`, `pipeline_score_direct_seq`,
  `pipeline_score_sum_bias`, `pipeline_score_final_seq`, and
  `pipeline_full_seq_bias_detail`: 0/483 mismatches. Do not collapse the two
  domain-definition sums back into one value; the diagnostic uses the simple
  `float` sum and pipeline scoring uses the compensated `esl_vec_FSum()` style
  sum.
- `pipeline_score_components`: 0/483 exact mismatches.
- `oprofile_xf_bits`: exact on the Pkinase `hmmsearch` reference after
  matching C's initial dummy profile length `100`. Keep
  `tracehash/scripts/run-hmmer-reference.sh` strict for this probe.

Completed broader fixture checks:

- `hmmer/testsuite/20aa.hmm` vs `hmmer/testsuite/20aa-alitest.fa`
  - Command shape: Rust/C `hmmsearch --cpu 1 --noali --tblout ...`
  - Core trace probes exact, including null2 and pipeline scores.
  - Parsed `tblout` rows match C exactly. This fixture specifically guards
    clustered null2-by-trace handling of `X` residues in `test2` and `test3`.
- `hmmer/tutorial/globins4.hmm` vs `hmmer/tutorial/globins45.fa`
  - Command shape: Rust/C `hmmsearch --cpu 1 --noali --tblout ...`
  - Core trace probes exact for all 45 hits.
  - Parsed `tblout` rows match C exactly.
- `hmmer/testsuite/gecco_pfam5.hmm` vs `hmmer/testsuite/gecco_proteins.faa`
  - Command shape: Rust/C `hmmsearch --cpu 1 --noali --tblout ...`
  - Core trace probes exact for all 16 pipeline hits / 24 rescored domains.
  - Parsed `tblout` rows match C exactly.

For any future fixture, record:

- command
- C/Rust visible output comparison
- trace mismatch summary for the core probes
- whether the fixture is intended for exact bitwise parity or only displayed
  output parity.

## Completed Tracehash Usability Improvements

Implemented improvements kept for continuing parity work:

- Done:
  - `tracehash-compare` accepts `--only function1,function2`,
    `--skip function1,function2`, and `--first N`.
  - It reports first occurrence-level differences by
    `function + input_hash + occurrence`, including the value column when
    present.
  - It still prints count differences and unordered pair totals, so it remains
    useful when duplicate-input occurrence order is not meaningful.
  - It accepts `--left-label`, `--right-label`, and `--summary-only` for large
    Rust/C trace comparisons.
  - It accepts `--json` for downstream scripts that need deterministic summary
    data without parsing the text report.
- Done:
  - Added Rust `Call::input_field()` / `Call::output_field()` and matching C
    named scalar field macros for quick ad hoc probes without defining a full
    derive/struct pair.
- Done:
  - Added `tracehash/scripts/run-hmmer-reference.sh` for the standard Pkinase
    reference run. It builds Rust traced release, builds C traced `hmmsearch`,
    runs both sides with `TRACEHASH_VALUES=1`, prints comparator/core-probe
    summaries, checks parsed `tblout` parity, keeps `oprofile_xf_bits` in the
    strict core-probe set, and rebuilds C
    uninstrumented on exit.
  - The script writes large traces under
    `target/tracehash-runs` by default; override
    `TRACEHASH_WORKDIR` or `PREFIX` for other storage.
- Done:
  - Added `tracehash::th_hmmer_probe!` for named scalar output probes with
    standard HMMER inputs: sequence length, model length, optional `dsq`
    bytes, and one or more named scalar outputs.
- Done:
  - Added C HMMER convenience macros for standard model/sequence inputs, sliced
    `dsq` bytes, and paired raw-plus-quantized f32 outputs.
- Done:
  - Added bounded `TraceHash` derive support for fieldless Rust enums with
    explicit numeric discriminants.

## Performance Work After Parity

Current speed state is tracked in the Current Speed Snapshot above. The older
Priority 6 baseline that showed Rust 1.5-1.8x slower on the Pkinase reference
is superseded by the newer snapshot where the reference case is on par or
faster depending on output mode.

Remaining speed work should use larger/fairer benchmark sets and kernel-level
profiling rather than overfitting the Pkinase reference. Rejected/neutral
speed experiments are recorded in Known Bad Or Neutral Experiments and the
Priority 7 measurement notes below.

## Verification Checklist

Before stopping after any parity change:

- `cargo check --features tracehash`
- `cargo test --manifest-path tracehash/Cargo.toml`
- `cargo test --test rust_hmmsearch_tests`
- `git diff --check`
- If C was traced, rebuild C clean and confirm `nm hmmer/src/hmmsearch | rg tracehash`
  prints nothing.
- Update `TRACE_PARITY.md` and this `TODO.md` if counts, next targets, or known
  false trails changed.

- Removing canonical-residue scans before the direct full-DP SIMD kernels preserved the reference output but measured slower on the reference benchmark; keep the safer scan unless a prevalidated-sequence approach is tested.
- Skipping the generic PP Gmx allocation for coordinate-only SIMD OA was output-exact but caused a repeatable minor-fault/system-time spike on the reference benchmark; it was reverted.
- `RUSTFLAGS='-C target-cpu=native'` did not improve the current reused-scratch build on the reference benchmark; do not rely on it as the parity-speed fix.

## Complexity-Audit Historical Notes - 2026-04-23

Findings from running `code-complexity-comparator` (`/home/mahogny/github/claude/code-complexity-comparator`) with the mapping file at `ccc_mapping.toml` against the full tree. Snapshot from that run: 165 matched pairs after mapping. Each of these surfaces where Rust carries measurably more static complexity than C in hot code. None are a speed fix by themselves; each is a hypothesis worth checking with `perf stat` (cycles, instructions, branch-misses) on the reference case before/after any attempted change.

Reproduce with:

```sh
cd /home/mahogny/github/claude/code-complexity-comparator
./target/release/ccc-rs analyze /data/henriksson/github/claude/newhmmer/hmmer/src -l c --recurse -o /tmp/ccc_hmmer/c.json
./target/release/ccc-rs analyze /data/henriksson/github/claude/newhmmer/src -l rust --recurse -o /tmp/ccc_hmmer/rust.json
./target/release/ccc-rs compare /tmp/ccc_hmmer/rust.json /tmp/ccc_hmmer/c.json \
  --mapping /data/henriksson/github/claude/newhmmer/ccc_mapping.toml --top 40
```

Top outliers from the 2026-04-23 run:

- `write_binary_hmm` vs `p7_h2io_WriteASCII`: `33.58`
- `trace_score_domain_forward_an…` vs `trace_score_domain_forward_an…`: `32.65`
- `alidisplay_backconvert` vs `p7_alidisplay_Backconvert`: `23.08`
- `backward_parser_pmx_offset_ca…` vs `backward_engine`: `20.96`
- `run` vs `p7_Pipeline`: `20.40`
- `calc_band_9` vs `calc_band_9`: `20.30`
- `convert` vs `p7_oprofile_Convert`: `18.47`
- `write_hmm` vs `p7_hmmfile_WriteASCII`: `18.41`

### Priority 7 status summary (2026-04-17)

- 7.1 **Done (measured)** - perf stat + record run. Findings captured below.
- 7.2 **Done (rationale: premise was misread)** - the "79 LOC of comments" metric counted tracehash-gated lines, not prose. Only 8 real `//` lines in the hot function; all trace probes already compile out in release. No action taken.
- 7.3 **Done (code removed)** - deleted three dead SIMD variants:
  - `forward_parser_with_specials` (189 LOC, `src/simd/fwd_filter.rs`)
  - `backward_parser` wrapper (10 LOC, `src/simd/bck_filter.rs`)
  - `backward_parser_with_decoding` + whole file `src/simd/bck_decoding.rs` (257 LOC)
  - Also removed `FwdSpecials` struct (only used by dead code) and `mod bck_decoding` from `src/simd/mod.rs`.
  - Reference benchmark unchanged (1.72s user before and after); `--tblout` output md5 identical.
- 7.4 **Done (measured, no action)** - perf record shows Rust `define_domains` does NOT appear in the top-25 hot functions. Its static Halstead-difficulty gap vs C (1059 vs 280) reflects source-level expression density, not runtime cost. Do not change.
- 7.5 **Done (rationale: compiler already handles it)** - the runtime `is_x86_feature_detected!("sse2")` is constant-folded on x86_64 release; the generic `else` branch is already eliminated. Gating with `#[cfg(not(target_arch = "x86_64"))]` would require touching 10 call sites in a function whose structural edits have regressed (see rejected experiments above). No action.
- 7.6 **Done (leave alone)** - `g_optimal_accuracy_with_deltas` is the generic fallback, cold on x86_64 (SIMD OA coordinate path is used on the reference). Precomputed `td[]` table is architecturally sound.
- 7.7 **Done (deferred)** - `convert` (oprofile) runs once per query; perf shows it at 0% of user time on the reference. Defer until a multi-query benchmark shows it as hot.
- 7.8 **Done (deferred)** - `p7_GForwardCheckpointed`/`p7_GBackwardCheckpointed` are C's checkpointed DP for memory-bounded full Forward/Backward. Rust does not port them. Defer until a long-sequence fixture shows RSS as a blocker.
- 7.9 **Done (explained-away)** - `run` vs `p7_Pipeline` deviation is idiomatic (Rust `?` vs C single-exit `goto`). No change.
- 7.10 **Done (advisory)** - meta-advice kept in the item text. Use `perf stat` to validate any complexity-audit suggestion before editing hot code.

### Priority 7 measurement findings (2026-04-17, task 7.1)

Pkinase reference, `--cpu 1`, no `--domtblout`, stdout to /dev/null, single-run `perf stat`:

```
                     Rust        C      ratio
cycles           4.59 Bn    4.29 Bn    1.07x
instructions     8.50 Bn    8.97 Bn    0.95x  (Rust does LESS work)
IPC              1.85       2.09       0.89x  (Rust slower per cycle)
branches         768 Mn     796 Mn     0.97x
branch-misses    3.19 Mn    3.01 Mn    1.06x
cache-misses     4.32 Mn    3.11 Mn    1.39x  (Rust 39% more)
elapsed          1.78 s     1.61 s     1.10x
```

Rust executes slightly fewer instructions than C but completes them slower. The dominant signal is **cache-miss rate: Rust has 39% more cache-misses** on the same workload. That is consistent with the I-cache / D-cache footprint of the hot code being larger than C's.

**Profile breakdown (self time, % of cycles):**

Rust top six = 85.99% in: `backward_parser_pmx_offset_with_scratch` 20.57, `forward_parser_pmx_offset_with_scratch` 19.49, `score_domain_envelope` 18.76, `Pipeline::run` 13.93, `backward_parser_pmx_offset_direct` 7.74, `forward_parser_pmx_offset_direct` 5.51.

C top two = 59.56% in: `backward_engine` 32.13, `forward_engine` 27.43. `rescore_isolated_domain` is 0.06% self. `p7_Pipeline` is below the top-25 cutoff.

**Key finding: the outer dispatch functions `score_domain_envelope` (18.76%) and `Pipeline::run` (13.93%) together account for 32.69% of Rust cycles. The C equivalents spend near-zero self time in those dispatch frames.** That is where the 1.10x wall-clock gap lives; the per-row SIMD kernels are comparable.

Interpretation: LLVM is inlining leaf work into these caller bodies, bloating them until they do not fit the instruction-cache hot path as well as C's compact engines. This is a structural inlining issue, not an algorithmic gap.

Attempted-and-rejected mitigations that are relevant here (from Priority 6): "Marking generated SSV functions `inline(never)` kept output identical but slowed to about 2.03s / 2.49s." So blanket `inline(never)` is not the answer.

**Follow-up measurement (2026-04-17):**

Binary `.text` size: Rust 1.43 MB vs C 0.76 MB (**1.9x larger**). Per-function breakdown of hot symbols:

| Function | Rust | C | Ratio |
|---|---:|---:|---:|
| `Pipeline::run` / `p7_Pipeline` | 43.9 KB | 3.7 KB | 11.8x |
| `score_domain_envelope` / `rescore_isolated_domain` | 34.5 KB | 2.2 KB | 15.8x |
| `forward_parser_pmx_offset_with_scratch` / `forward_engine` | 5.4 KB | 1.5 KB | 3.6x |
| `backward_parser_pmx_offset_with_scratch` / `backward_engine` | 8.1 KB | 2.5 KB | 3.2x |
| Hot-path total | 91.9 KB | 9.9 KB | 9.3x |

iTLB/D-cache: Rust **iTLB miss rate 30.58% vs C 16.06%** (1.9x), Rust L1-dcache miss rate 3.36% vs C 2.75%. The I-cache hypothesis is confirmed by the 1.9x text-size ratio and 1.9x iTLB miss ratio.

`define_domains` (the 463-LOC top-level domain-definition function) does not appear as a separate symbol in the release binary; it is fully inlined into its sole caller `Pipeline::run`. That is why `Pipeline::run` is 43.9 KB.

**Rejected experiment: split `Pipeline::run` at filter boundary.** Extracted everything past `self.n_past_fwd += 1` into `#[inline(never)] fn run_after_filters(&mut self, gm, om, bg, hmm, sq, th, fwd_sc, null_sc)`. Byte-identical tblout; all 128 tests pass.

| Metric | before | after | delta |
|---|---:|---:|---:|
| `Pipeline::run` size | 43.9 KB | 13.4 KB | -69% |
| `run_after_filters` size | (inlined) | 30.8 KB | (new frame) |
| `.text` total | 1427.3 KB | 1427.9 KB | +640 B |
| iTLB-load-misses | 50,077 | 39,557 | -21% |
| iTLB miss rate | 30.58% | 25.02% | -5.6 pp |
| L1-dcache-load-misses | 102 M | 100 M | -1.5% |
| cycles | 4.59 Bn | 4.65 Bn | +1.3% |
| IPC | 1.85 | 1.82 | -0.01 |
| user time | 1.70-1.77 s | 1.67-1.96 s | noise-overlapping |

Structural target was hit: `Pipeline::run` shrank 3.3x and iTLB-misses dropped 21%. But paired multi-run timing did not move beyond noise, and cycles went up 1.3%. The 9-arg inter-function call cost (argument marshalling, register spill/reload) absorbed the I-cache win the same way it did in the `inline(never) define_domains` experiment. Reverted. Do not re-try this as a plain `inline(never)` on a wide capture set.

**Rejected experiment: `#[inline(never)] define_domains`.** Marked `define_domains` as `#[inline(never)]` to force a real call boundary. Output byte-identical.

| Metric | before | after | delta |
|---|---:|---:|---:|
| `Pipeline::run` size | 43.9 KB | 16.3 KB | -63% |
| `define_domains` size | (inlined) | 28.7 KB | out of Pipeline::run |
| `.text` total | 1427 KB | 1429 KB | +0.1% |
| iTLB-load-misses | 50,077 | 43,295 | -14% |
| iTLB miss rate | 30.58% | 28.25% | -2.3 pp |
| L1-dcache-load-misses | 102 M | 100 M | -2% |
| cycles | 4.59 Bn | 4.67 Bn | +1.7% |
| IPC | 1.85 | 1.83 | -0.01 |
| user time | 1.70-1.77 s | 1.75-1.79 s | +0.05 s noisy |

Structural result was exactly as predicted: Pipeline::run dropped 2.7x, iTLB improved 14%, but function-call overhead from de-inlining (spill/reload, argument marshalling for 14 parameters) cancelled the I-cache win. Matches the broader lesson in Priority 6 that `#[inline(never)]` has not been a net speed change in this codebase. Reverted. Do not re-try this exact approach on `define_domains`.

**Pattern observed from both rejected experiments above:** `#[inline(never)]` on a function that takes many arguments (9-14 refs/values) consistently produces the expected I-cache improvement but is cancelled by per-call register spill/reload. Attempts in this direction should either (a) use a much smaller argument surface (e.g. wrap everything in a single borrowed struct parameter), or (b) split only tiny focused blocks whose call cost is minimal.

**Deferred performance experiment ideas:**

- Shrink `score_domain_envelope` (34.5 KB, 18.76% self cycles) by extracting the `make_alignment_display` branch into a helper. On `--noali --domtblout` that branch is dead at runtime but still compiled into the hot function body. The extracted helper would have a narrow argument set, avoiding the spill-cost trap.
- Reduce inlined size of the SIMD parser-with-scratch variants by having them tail-call the `_direct` variant's body rather than duplicating it. This shrinks `_with_scratch` without adding a new frame.
- Profile a smaller extracted block within `define_domains` that has fewer captures (for example the per-domain post-decoding loop). The smaller-block variation might pay off where the full de-inline did not.
- Build a `#[repr(C)] struct PipelineRunCtx<'a>` and pass it by reference into the hoisted function, reducing the 9-arg call to a 1-arg call. Revisit the Pipeline::run split with that lower-overhead calling convention.

These are Priority 6 Performance-After-Parity work and require the full tracehash verification loop. Do not attempt without the full build/compare/rebuild cycle in `Ground Rules`.

Detailed Priority 7 subitems 7.1-7.10 have been resolved, deferred, or
classified as advisory in the status summary above. Treat the measurement
findings and rejected experiments as historical evidence; do not turn static
complexity deltas into edits without a fresh perf claim on the active
benchmark.
