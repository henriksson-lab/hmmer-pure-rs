These fixtures are self-consistent `--mapali` inputs.

The legacy HMMER testsuite ships several `.hmm` files whose stored `CKSUM`
no longer matches the paired Stockholm alignment. That is useful as a negative
test for upstream-compatible checksum rejection, but it cannot be used for
positive `--mapali` coverage.

The rebuilt `.hmm` files in this directory are generated from the checked-in
Stockholm sources with the current Rust `hmmer build` implementation:

- `20aa-rebuilt.hmm` from `hmmer/testsuite/20aa.sto`
- `Caudal_act-rebuilt.hmm` from `hmmer/testsuite/Caudal_act.sto`
- `ecori-rebuilt.hmm` from `hmmer/testsuite/ecori.sto` with `--dna`

They intentionally do not replace the legacy testsuite HMMs in place, because
those older files are still referenced by unrelated search parity tests.
