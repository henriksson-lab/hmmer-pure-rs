# Trace Parity Notes

This file records what the Rust port has actually been compared against C
HMMER with `tracehash`. Keep entries narrow: a function is only "matched" for
the probe surface and test case listed here.

## Current Reference Case

- HMM: `test_data/Pkinase_pfam.hmm`
- Sequences: `test_data/human_swissprot_2k.fasta`
- Command shape:
  - Rust: `TRACEHASH_VALUES=1 target/release/hmmer search --cpu 1 ...`
  - C: `TRACEHASH_VALUES=1 hmmer/src/hmmsearch --cpu 1 ...`

## Matched Or Improved

- Pipeline filter counts match C for the reference case:
  - MSV: 1229
  - bias: 909
  - Viterbi: 519
  - Forward: 483
- `Bg::filter_score()` was aligned to C `esl_hmm_Forward()` behavior.
- SIMD Forward optimized profile conversion now uses an Easel
  `esl_sse_expf()`-style approximation for `rfv` and `tfv`, matching C's
  profile odds conversion more closely than Rust `f32::exp()`.
- Isolated-domain Forward now clones and reconfigures the optimized profile
  into unihit mode, matching C `p7_oprofile_ReconfigUnihit()` instead of
  rebuilding through generic log-space scores.
- Isolated-domain Forward row anchors match for common envelopes through row16
  at `1e-5` quantization in the reference case.
- Row15/k15 recurrence probes match C bitwise for the current reference case
  after optimized-profile reconfiguration parity.
- SIMD Backward DD wing unfolding now carries C's last DD contribution between
  serialized passes instead of the accumulated D cell. On the reference trace
  this reduced `score_domain_oa` mismatches from 549 to 29, isolated-domain
  Forward missing inputs from 76 to 26, and `simd_backward_states_q1e5_summary`
  mismatches from 483 to 94.
- SIMD Backward row probes show row L matches at `1e-5`; the remaining
  Backward drift appears after recursion/termination (`row0`: 3 mismatches,
  `row1`: 4 mismatches on the reference case). Making Backward's
  `has_own_scales` overflow mode sticky, as in C, was behaviorally neutral on
  this reference trace but keeps the rare overflow path faithful.
- Backward component probes narrow the earliest remaining full-sequence
  Backward drift: row0 differs only through `B`/`N`, while row1 `B` and scale
  match at `1e-5` and the row1 drift is in propagated `E/J/N`. Engine-level
  row1 rolling-DP probes show row1 M/D/I sums still have small `1e-5`
  mismatches across the common full-sequence calls, so the next Backward target
  is the rolling row recurrence or its scale edge, not domain-definition code.
- Parser-only C engine probes now match Rust probe counts for row1
  (`483`/`483`). Row1 phase checkpoints show the row1 M/D/I mismatch is already
  present after phase 1, before special-state propagation, DD extension, M->D,
  or row scaling. Therefore the next Backward trace target is the previous
  row/input used by phase 1, especially row2/row4 recurrence drift.
- Sparse parser Backward engine anchors now cover both absolute rows
  (`row128`..`row1`) and end-relative rows (`rowL-1`, `rowL-2`, `rowL-4`,
  `rowL-8`, `rowL-16`, ...). In the reference trace, rowL, rowL-1, rowL-2,
  rowL-4, and rowL-8 match at `1e-5`; the first observed end-relative mismatch
  is a single `rowL-16` M-sum mismatch. Absolute rows 128..1 still show small
  M/D/I sum differences, so the drift accumulates after the first few
  recurrences rather than starting in rowL initialization.
- Backward phase 1 in Rust now reads the previous-row M stripe before storing
  the current-row M stripe, matching C parser read/write order. This was neutral
  on the current trace but keeps the separate-buffer Rust implementation closer
  to C's in-place parser.
- Main `hmmsearch` hit-table reporting now selects the best domain by domain
  `lnP` instead of always printing `dcl[0]`. This fixed the visible large
  best-domain score discrepancy in the reference output (for example
  `KS6A1_HUMAN` now reports the 267.8-bit second domain, matching C, instead of
  the 241.9-bit first domain).
- Main hit tables now print C's reported-domain count (`nreported`) in the `N`
  column instead of raw `ndom`, and the detailed `hmmsearch` domain/alignment
  sections skip unreported domains. This eliminated the previous visible
  domain-count reporting mismatches in the reference table.
- `tracehash` now has an opt-in `TRACEHASH_VALUES=1` debug column for scalar
  probe values and byte-slice summaries. The normal comparison key remains
  hash-only and existing TSV readers remain compatible.
- Clustered-region stochastic traceback in Rust now has an optimized-profile
  probability-space path for SSE, matching C's `p7_StochasticTrace()` more
  closely than the generic log-space traceback.
- Stochastic traceback now uses Easel's `esl_randomness_CreateFast()` stream
  (legacy Knuth LCG) rather than MT19937. This fixed the previously visible
  `E2AK4_HUMAN` clustered envelope start for region 323..539 (`325..539` in
  both Rust and C on the reference case).
- Segment-pair clustering now follows Easel's stack-based
  `esl_cluster_SingleLinkage()` traversal instead of union-find/`HashMap`
  collection. On the reference case, `spensemble_cluster_candidate`,
  `domain_cluster_summary`, and `domain_envelope_candidate` trace values now
  match C exactly.
- `Trace::append()` now canonicalizes coordinates like C `p7_trace_Append()`:
  nonemitting states have `i=0`, and only repeated `N/C/J` states carry an
  emitted residue coordinate. This fixed hidden stochastic traceback
  differences in clustered-region null2 sampling. On the isolated
  `E2AK2_HUMAN` reference case, `region_null2_trace_segment` now matches C for
  all 276 sampled domain segments, and the domain null2 correction matches C
  bitwise.
- Rust now mirrors C `impl_Init()` for x86 floating point mode: the CLI,
  `p7_flogsuminit()`, and Rayon search workers enable SSE flush-to-zero and
  denormals-are-zero. On the reference trace this reduced
  `region_forward_summary` hash mismatches from 11 to 7 and
  `simd_forward_specials_summary` mismatches from 483 to 471 without changing
  the displayed main hit table.
- `Bg::null_one()` now matches C `p7_bg_NullOne()`'s arithmetic: `p1` remains
  `float`, but the logarithms and multiply-add are evaluated in double
  precision before the final cast to `float`. On the reference trace this
  reduced exact `pipeline_msv_decision` mismatches from 5526 to 649 and
  `pipeline_score_components` mismatches from 359 to 340 without changing the
  displayed main hit table.
- Pipeline P-value conversion now matches C's cast points: subtract in `float`,
  divide by the double `eslCONST_LOG2`, then store the bit score as `float`
  before calling the distribution survival function. Together with matching
  C's bias-filter trace semantics for MSV-overflow cases, this brings
  `pipeline_msv_decision` to 0 mismatches and reduces
  `pipeline_bias_decision` to 20 remaining one-ulp filter-score/P-value
  mismatches on the reference trace.
- Bias-filter HMM scoring now follows C `esl_hmm_Forward()` more closely:
  per-row scale logs use double `log()` with a final `float` cast, and filter
  emission odds include Easel's degenerate residue codes (`B/J/Z/O/U/X` for
  amino and ambiguity codes for DNA/RNA). This fixes the previous
  `SELK_HUMAN` selenocysteine (`U`) filter-score mismatch. On the reference
  trace, `pipeline_msv_decision`, `pipeline_bias_decision`, and
  `pipeline_vit_decision` now all have 0 mismatches.
- Post-Forward sequence score conversions now use the same nats-to-bits cast
  point as C (`float` subtraction, double `eslCONST_LOG2` division, final
  `float` cast), and reconstruction length corrections use double `log()` like
  C. This reduced `pipeline_score_final_seq` mismatches from 209 to 200 on the
  reference trace; remaining score-component drift is dominated by Forward and
  domain-rescoring inputs.
- The C trace build helper now rebuilds `hmmer/src/impl_sse/fwdback.o` with
  `TRACEHASH`, so hot SIMD Forward probes are present in fresh C traces.
- SIMD domain null2 now uses a striped expectation helper in the no-alignment
  path, matching C `p7_Null2_ByExpectation()`'s vector-style accumulation more
  closely than the generic posterior-matrix shortcut. On the current
  `TRACEHASH_VALUES=1` reference, `score_domain_null2` and
  `define_domains_summary` both have 0 exact mismatches, restoring null2 parity
  while keeping the corrected C SIMD Forward trace surface.
- Full-PMX SIMD Forward now reconstructs its returned score from the stored
  row-scale array after the DP matrix is filled, matching C's final score
  accumulation without perturbing the hot recurrence. With the corrected C trace
  surface, `pipeline_score_fwd_sc` is now 0/483 mismatches, `score_domain_forward`
  is down to 6/586 exact mismatches, and `pipeline_score_final_seq` is down to
  3/483 exact mismatches on the `--noali` reference case. A later
  running-`totscale` return experiment made `score_domain_forward` exact but
  perturbed domain-null2/define-domain parity, so it was reverted.
- `tracehash` Rust quantized-float hashing now matches the C helper's `float`
  divide plus add/subtract `0.5f` then truncate rule. This removed false
  `pipeline_score_fwd_sc` mismatches where raw float bits were already equal.
- C pipeline score-component probes now preserve the actual full-sequence bias,
  direct sequence score, and reconstruction score while they are live, instead
  of recomputing or round-tripping them after bit-score conversion. This reduced
  `pipeline_score_components` mismatches on the current `TRACEHASH_VALUES=1`
  reference comparison from 144 to 25.
- Domain definition now returns both C sequence-bias sums: a simple `float` sum
  for the `define_domains_summary` diagnostic, and an `esl_vec_FSum()` style
  compensated sum for pipeline scoring. The new
  `pipeline_full_seq_bias_detail` probe matches C at 0/483, and
  `pipeline_score_full_seq_bias`, `pipeline_score_direct_seq`, and
  `pipeline_score_final_seq` are all 0/483 on the reference case.
- Clustered-region isolated-domain rescoring now runs full-PMX Forward even
  when Rust does not need the matrix later, matching C `rescore_isolated_domain`
  calling `p7_Forward()` with `ox1`. This closes the last known exact score
  drift on the reference trace: `score_domain_forward`,
  `pipeline_score_sum_nats`, `pipeline_score_sum_bits`, and
  `pipeline_score_components` are all 0 mismatches.
- Null2 correction now applies C-style alphabet degeneracy handling when scoring
  residues from null2 odds: canonical residues use their own odds, ambiguous
  residue codes average canonical odds, and gap/nonresidue/missing stay neutral.
  This closes a semantic gap for ambiguous residues even though the current
  Pkinase reference counts were unchanged.
- Clustered-region null2-by-trace now applies the same degenerate-residue odds
  averaging before the final per-residue log conversion. This fixed the
  `hmmer/testsuite/20aa.hmm` vs `20aa-alitest.fa` fixture, where `test2` and
  `test3` contain `X` residues: all core trace probes are exact there except
  the same benign conversion-time `oprofile_xf_bits` 6/8, and the parsed
  `tblout` rows match C exactly.
- The `hmmer/tutorial/globins4.hmm` vs `globins45.fa` fixture is exact for all
  core trace probes across 45 hits, again with only the benign
  conversion-time `oprofile_xf_bits` 6/8 mismatch. Parsed `tblout` rows match C
  exactly.
- The `hmmer/testsuite/gecco_pfam5.hmm` vs `gecco_proteins.faa` fixture is
  exact for all core trace probes across 16 pipeline hits and 24 rescored
  domains. The only probe mismatch is the same benign conversion-time
  `oprofile_xf_bits`, now 30/40 across the five HMMs. Parsed `tblout` rows
  match C exactly.
- The reference `hmmsearch` main hit table now matches C at displayed precision:
  483 Rust rows, 483 C rows, identical hit set, and zero differing main-table
  rows after parsing.

## Known Remaining Differences

- After optimized-profile reconfiguration parity, isolated-domain Forward row
  anchors match for common envelopes through row16. Row17 still has a small
  `q1e-5` special-state mismatch, but row17 M lane0 per-`q` exact-bit probes
  match for all common input hashes; the remaining lane0 aggregate mismatch is
  most likely trace aggregation/quantization boundary behavior or a consequence
  of divergent envelope populations, not a row17 recurrence mismatch.
- Matching C operand order in the Forward D->D `_mm_add_ps()` sites did not
  change the row17 trace counts.
- Full-sequence SIMD Forward still diverges before downstream domain scoring
  (`simd_forward_row19_q1e5`, row128/rowL, and scale-event summaries), making it
  a likely source of some downstream drift. Backward parser row anchors now
  show the remaining Backward recurrence drift starts only after several
  matching rows from the end of each target.
- The two `simd_forward_row19_e_q1e5` mismatches map to canonical kinase
  targets `CDK1_HUMAN` and `CDK5_HUMAN`. Engine-level row19 probes show exact
  `xEv` lane bits and final `xE` bits match C for those two inputs; CDK1 also
  matches row18/row19 M/D sums at `1e-5`, while CDK5 only shows a row19 M-sum
  edge. This suggests the remaining row19 special mismatch is not an obvious
  row19 recurrence error in the live Forward engine.
- Forward sparse-rescale scalar specials now use C's division form (`xN/xE`,
  etc.) instead of multiplying by a precomputed reciprocal. This removed the
  `simd_forward_engine_first_scale_*` and `simd_forward_first_scale_row_q1e5`
  mismatches on the reference trace. The remaining `simd_forward_scales_summary`
  mismatches map to later scale-edge cases (`LATS1_HUMAN`, `STK38_HUMAN`), not
  the first scale event.
- Wider Forward scale-event instrumentation now keys rescale events by
  parser/full mode and event index. For the two remaining full-sequence scale
  array mismatches, the row is the same in Rust and C but the event-10 `xE`
  bits differ: `LATS1_HUMAN` event 10 stores row 919, and `STK38_HUMAN` event
  10 stores row 292. Events before 10 match for these inputs. The next Forward
  target is therefore the recurrence immediately before event 10, not scale
  storage or first-scale special-state division.
- Event-10 window probes narrow the Forward drift further. `LATS1_HUMAN`
  matches through row 918 and first differs at row 919; `STK38_HUMAN` matches
  at row 280 and first differs by row 291. At those first-bad rows, row-start
  `xB`, `xJ`, and `xC` bits match, while `xN` bits differ. The main recurrence
  already has M/D differences before DD unfolding, while I matches. C assembly
  for the traced build uses scalar single-precision `divss` for the sparse
  rescale reciprocal, so Rust uses `1.0f32 / xE`; trying an f64 reciprocal did
  not remove the event-10 drift and was less faithful to the compiled C binary.
- Exact row-bit hashes at the event-10 windows and immediately after sparse
  rescale are intentionally too sensitive for this case: they differ even when
  q1e-5 sums match. Use them to prove bitwise nonidentity, not to localize the
  visible output discrepancy.
- C `p7_BackwardParser()` only keeps a rolling main-state row, so DP-row
  Backward probes must live inside the Backward engine. Probing `oxb->dpf[1]`
  later in domain definition is invalid for parser matrices.
- Downstream null2/OA/domain outputs still diverge because they depend on the
  remaining Forward scale-edge cases, Backward q1e5 differences, and posterior
  decoding/null2 behavior.
- SIMD domain decoding now mirrors two C cast/order details: the Backward scale
  product update is conditional on `has_own_scales`, and `mocc` uses a `float`
  `njcp` accumulator followed by C's `1. - njcp` cast point. The `njcp`
  accumulator change reduced `domain_decoding_summary` mismatches from 483/483
  to 249/483 on the corrected reference trace.
- Backward parser sparse-rescale now divides scalar special states by the row
  scale, matching C, while still using a reciprocal vector for DP-row scaling.
  This first reduced `domain_decoding_summary` to 207/483, `score_domain_null2`
  to 2/586, and `define_domains_summary` to 7/483 on the corrected reference
  trace.
- Optimized-profile instrumentation now covers generic profile entry scores,
  source `tfv` log-score vectors, final `tfv` probability vectors, and `rfv`
  emission vectors. The reference mismatch in `tfv[161]` lane 2 was caused by
  Rust's occupancy recurrence using pure `f32` arithmetic for
  `(1.0 - mocc[k-1])`; C's `1.0` literal promotes that part of the expression
  to double. Matching that mixed-precision recurrence made
  `profile_entry_source_bits`, `oprofile_tfv_source_bits`, `oprofile_tfv_bits`,
  and `oprofile_rfv_bits` exact on the reference case.
- After the occupancy fix, the broad downstream traces that previously pointed
  at Forward/Backward/posterior drift are exact on the reference case:
  `forward_special_step_bits` 0/10233, `domain_decoding_summary` 0/483,
  `score_domain_null2` 0/586, `score_domain_forward` 0/586,
  `define_domains_summary` 0/483, `pipeline_score_final_seq` 0/483, and
  `pipeline_score_components` 0/483. The only known remaining mismatch in the
  core summary is conversion-time `oprofile_xf_bits` 6/8 for overwritten
  `N/C/J` specials. The hit table remains identical on the parsed reference
  output.
- The conversion-time `oprofile_xf_bits` mismatch is accepted as benign for the
  checked search paths. The mismatching `N/C/J` odds are overwritten by
  optimized-profile reconfiguration before scoring; `E` specials match. A
  direct Rust call to C `expf()` for these conversion values did not close the
  trace and caused broad downstream null2/domain hash drift, so preserve the
  current conversion unless a new traced path proves these values are live.
- The reference workflow is now scripted in
  `tracehash/scripts/run-hmmer-reference.sh`. Final run on the Pkinase
  reference case matched 307492 Rust rows to 307492 C rows when skipping only
  benign `oprofile_xf_bits`; every core score/domain probe was exact and parsed
  `tblout` matched for 483 rows. Future large trace outputs should use the
  script default under `target/tracehash-runs`, not external scratch paths.
- A temporary `score_domain_null2_odds` probe showed that simple-domain null2
  drift came from using the generic-layout null2 helper when `--domtblout`
  forced alignment coordinate calculation. C still computes null2 with
  optimized-profile `p7_Null2_ByExpectation()`. Rust now uses the optimized OMX
  null2 helper for expectation even when it also builds a generic posterior
  matrix for alignment coordinates.

## Comment Policy

Use local Rust comments only for durable facts that affect implementation, such
as "C uses `esl_sse_expf()` here". Use this file for broader trace status, test
case notes, and "verified so far" claims.
