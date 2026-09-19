# Known deviations from upstream HMMER

This file records places where hmmer-pure-rs deliberately does **not**
reproduce the behaviour of the tracked upstream commit
(`EddyRivasLab/hmmer@9acd8b6758a0`). Each entry says what C does, what this
port does, and why the difference is kept. Unintended divergences belong in
`TODO.md`, not here.

## hmmbuild: `-n` is honoured under `--singlemx`; C ignores it

**Upstream behaviour.** `-n <name>` renames the input alignment before the
build (hmmbuild.c:1361-1363). For a normal multi-sequence build `p7_Builder`
copies the alignment name into the model, so `-n` works. Under `--singlemx`,
hmmbuild instead fetches the single sequence out of the alignment and hands it
to `p7_SingleBuilder`, which names the model after the *sequence*
(seqmodel.c:88). The renamed alignment is never consulted, so `-n` reaches the
summary table (which prints `msa->name`, hmmbuild.c:1294) but not the `NAME`
line of the written model:

```sh
$ hmmbuild -n foo --singlemx out.hmm single.sto   # single.sto's one sequence is "q1"
1     foo   1    22    20     1.00  0.660          # table shows foo ...
$ grep ^NAME out.hmm
NAME  q1                                          # ... but the model is still q1
```

**This port.** The summary table prints `foo`, as C does. The model is also
named `foo`: `-n` is applied to the finished model on every build path, so
`NAME  foo` is written. A model built with `-n --singlemx` therefore differs
from C's by its `NAME` line (and anything derived from it, such as `#=GF ID`
in `-A` output downstream).

**Why it is kept.** The C behaviour looks like an oversight rather than a
design choice: `-n` is documented as "name the HMM", and nothing in upstream
depends on `--singlemx` ignoring it. Reproducing it would make the option
silently do nothing in one mode. Anyone who needs a byte-identical C model
here can rename the sequence in the input instead of using `-n`.

Everything else about `--singlemx` naming matches C, including the sequence
name being used (not the alignment name) when `-n` is absent; see
`hmmbuild_singlemx_builds_one_sequence_model_with_gap_options` in
`tests/cli_output_parity.rs`.

## Written models carry no `DATE` line

**Upstream behaviour.** Every build path calls `p7_hmm_SetCtime()`
(seqmodel.c:91, p7_builder.c) and `p7_hmmfile_WriteASCII` then emits
`DATE  <ctime>`, so two otherwise identical builds differ by their timestamp.

**This port.** `ctime` is never set on any build path, so no `DATE` line is
written. Models read from a file keep whatever `DATE` they came with.

**Why it is kept.** Byte-reproducible output is what the golden-file and
C-parity tests depend on; every C comparison in `tests/` already strips
`DATE`. Adding the line would make each fresh build unique for no analytical
benefit. Decision recorded 2026-09-19.

## Score-system errors are reported before input-file errors

**Upstream behaviour.** hmmbuild, phmmer, jackhmmer and nhmmer open their
input files first and only then call `p7_builder_{Load,Set}ScoreSystem()`
(e.g. nhmmer.c:892-894, hmmbuild.c:595-597). With both a bad `--mx` and a
missing input file, C reports the file error.

**This port.** The score system is validated up front, so the same command
line reports the `--mx`/`--mxfile` error. The message text matches C; only
the precedence differs, and only when two errors are present at once.

**Why it is kept.** Error precedence between independent failures is not
part of the translation's scope. Decision recorded 2026-09-19.
