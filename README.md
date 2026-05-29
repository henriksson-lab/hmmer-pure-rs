# hmmer-pure-rs 0.7.2

A Rust port of [HMMER 3.4](http://hmmer.org/) for biological sequence analysis using profile hidden Markov models (profile HMMs). Searches sequence databases for homologous sequences.

Original-code snapshot used for translation/parity work: HMMER git commit `9acd8b6758a0ca5d21db6d167e0277484341929b`.

* 2026-05-29: Further regressions fixed. Expecting more to land
* 2026-05-28: A slur of further edits have landed. More testing to be done but audits have converged for now
* 2026-05-26: Features now appears to be in. Initial testing suggests parity but **further audit is likely needed**
* 2026-05-23: **New audit strategy has uncovered a large number of problems, being fixed.**
* 2026-04-27: The code has passed current methods of testing and **careful use on real data is possible**. Treat performance claims as workload-specific and rerun the real-world benchmark harness before relying on them.


## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation


## Benchmarks

Do not treat static README timing numbers as portable evidence. Benchmarks are
faithful only when they are rerun against realistic fixtures, compared with the
bundled C HMMER snapshot, and checked for output agreement.

The reproducible search benchmark entry point is:

```bash
scripts/benchmark_real_world.sh
```

These harnesses are intended for a source checkout. Release packages include
them for provenance, but the realistic external fixtures must still be prepared
locally, and `C_HMMER_DIR` must point at a built C HMMER tree when `hmmer/src`
is not present.

The script:

- builds `target/release/hmmer` unless `SKIP_BUILD=1` is set
- runs Rust and bundled C HMMER on medium/large UniProt fixtures under
  `external/`
- benchmarks representative `hmmsearch`, `phmmer`, `jackhmmer`, `hmmscan`,
  `nhmmer`, and `nhmmscan` workloads
- records command lines, dataset sizes, wall/user/system time, max RSS, table
  row counts, top-hit ordering, selected cases, run order, generated-query
  source checksums, git commit, git status/diff audit files, `.cargo/config.toml`
  content/checksum or absence, Rust/C executable metadata, and host metadata
- validates `CASES` selectors and fails on unknown or zero selected cases
- alternates Rust/C execution order by round to reduce one-sided warm-cache
  effects, and records the order in `run_order.tsv`
- fails by default if Rust and C disagree on `tblout`/`domtblout` row counts,
  normalized core table rows, normalized full outputs, or top `tblout` target
  ordering
- writes artifacts to `reports/benchmarks/<timestamp>/`

Utility timing claims use a separate checked harness:

```bash
scripts/benchmark_utilities.sh
```

That script runs the README utility cases (`hmmbuild`, `hmmalign`, `hmmstat`,
`hmmconvert`, `hmmlogo`, `hmmemit`, `hmmsim`, `alimask`, `makehmmerdb`,
`hmmpress`, and `hmmfetch`) against bundled C HMMER. It records the same basic
timing/host metadata plus input sizes and SHA-256 checksums, and it compares
normalized outputs for cases where C-equivalent text output is expected.
`hmmsim` and `makehmmerdb` are timed/status-checked but not normalized-output
compared because their remaining formatting/container differences are listed
under Remaining Gaps.

Useful controls:

```bash
THREADS=4 ROUNDS=3 scripts/benchmark_real_world.sh
CASES=hmmsearch_human_pkinase SKIP_BUILD=1 scripts/benchmark_real_world.sh
CASES=hmmsearch_human_pkinase,phmmer_human_cyh3 ALLOW_MISMATCH=1 scripts/benchmark_real_world.sh
SKIP_BUILD=1 OUT_DIR=reports/benchmarks/manual scripts/benchmark_real_world.sh
CASES=hmmstat_gecco_cluster1,hmmfetch_gecco_cluster1_valid scripts/benchmark_utilities.sh
```

To smoke-test every Rust subcommand on a separate set of newly gathered real
fixtures, first download the fixtures and then run the all-tools harness:

```bash
scripts/download_new_real_world_fixtures.sh
scripts/benchmark_new_real_all_tools.sh
```

That workflow uses E. coli K-12 UniProt/RefSeq data plus extracted Pfam/Rfam
seed alignments, keeps large files under ignored `external/new_real/`, and
writes command/status artifacts under `reports/benchmarks/`.

For a heavier, more realistic eukaryotic workload, use:

```bash
scripts/download_realistic_fixtures.sh
scripts/benchmark_realistic_all_tools.sh
scripts/benchmark_realistic_compare_c.sh
```

That workflow uses the S. cerevisiae UniProt proteome, full Ensembl yeast
genome FASTA, a multi-model panel extracted from the current Pfam-A HMM
archive, and Pfam/Rfam seed alignments. It writes fixtures under ignored
`external/realistic/`. The comparison harness runs matching Rust and bundled C
HMMER commands and records wall time, user/system time, and maximum RSS.

See `REAL_WORLD_FIXTURES.md` for dataset sources and fixture layout. Small
checked-in fixtures remain useful for parity tests and smoke tests, but they
are not representative performance evidence.

### Benchmark Results

This README intentionally does not publish static timing tables. Benchmark
reports are ignored/generated artifacts and are only auditable together with
their `metadata.txt`, `results.tsv`, `datasets.tsv`, `run_order.tsv`, command
files, checksums, and git audit files. Regenerate local search and utility
tables with the harnesses above when you need current speed/RSS evidence.

## Features

- Pure Rust implementation of the HMMER search pipeline
- SSE2-accelerated MSV, Viterbi, and Forward filters
- Full domain definition with posterior decoding (btot/etot/mocc region detection)
- Stochastic traceback clustering for multi-domain sequences
- Null2 bias correction with omega weighting
- Composition bias filter matching C HMMER
- Reads modern HMMER3 ASCII HMM files and C `.h3m` binary model records
- Reads FASTA and gzipped FASTA (.fasta.gz) sequence databases
- Core C HMMER `hmmsearch` flags are implemented (`--cut_ga`, `-Z`,
  `--nobias`, `--acc`, etc.); unsupported options are rejected rather than
  silently ignored
- Search tabular outputs (`--tblout`, `--domtblout`, `--pfamtblout`) are
  covered by parity tests on representative fixtures
- Library API for programmatic use without file I/O

## Build

```bash
cargo build --release
```

For best performance, compile with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## CLI Usage

All tools are accessed as subcommands of the `hmmer` binary:

```bash
# Search HMM(s) against a sequence database
hmmer search query.hmm sequences.fa
hmmer search --tblout hits.tbl query.hmm sequences.fa
hmmer search --cpu 4 -E 0.001 query.hmm sequences.fa
hmmer search --cut_ga query.hmm sequences.fa       # Pfam gathering cutoffs
hmmer search -Z 10000 query.hmm sequences.fa       # set database size
hmmer search --acc --noali query.hmm sequences.fa   # accession names, no alignments
hmmer search query.hmm sequences.fa.gz              # gzipped FASTA

# Build HMM from alignment
hmmer build output.hmm alignment.sto

# Search sequence against HMM database
hmmer press -f hmm_database.hmm
hmmer scan hmm_database.hmm query.fa

# Protein sequence vs database (builds HMM on the fly)
hmmer phmmer query.fa database.fa

# Iterative search
hmmer jackhmmer -N 3 query.fa database.fa

# DNA/RNA search
hmmer nhmmer query.hmm dna_target.fa

# Utility commands
hmmer stat model.hmm
hmmer emit -c model.hmm
hmmer convert model.hmm
hmmer fetch database.hmm "model_name"
hmmer align model.hmm sequences.fa
hmmer logo model.hmm
hmmer pgmd --hmmdb database.hmm --port 51371
hmmer pgmd --seqdb proteins.fa --port 51371
```

`hmmer convert` reads and writes HMMER3 ASCII/binary profiles, and can write
HMMER2 ASCII output with `-2` for C `hmmconvert` compatibility. Reading HMMER2
input remains unsupported.

### hmmsearch flags

```
Output:       -o, --tblout, --domtblout, --pfamtblout, -A, --noali, --acc, --notextw, --textw
Thresholds:   -E, -T, --domE, --domT, --incE, --incT, --incdomE, --incdomT
Cutoffs:      --cut_ga, --cut_nc, --cut_tc
Filters:      --max, --F1, --F2, --F3, --nobias
Expert:       --nonull2, -Z, --domZ, --seed, --tformat fasta, --cpu
```

## Library Usage

```rust
use hmmer_pure_rs::{Alphabet, Bg, Pipeline, Profile, OProfile, TopHits};
use hmmer_pure_rs::hmmfile;
use hmmer_pure_rs::profile::{profile_config, reconfig_length, P7_LOCAL};
use hmmer_pure_rs::sequence::Sequence;
use std::path::Path;

// Load an HMM
let hmms = hmmfile::read_hmm_file(Path::new("query.hmm")).unwrap();
let hmm = &hmms[0];

// Set up alphabet, background model, and scoring profile
let abc = Alphabet::new(hmm.abc_type);
let bg = Bg::new(&abc);
let mut gm = Profile::new(hmm.m, &abc);
profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
let om = OProfile::convert(&gm);

// Create pipeline and hits collector
hmmer::logsum::p7_flogsuminit();
let mut pli = Pipeline::new();
pli.new_model(&gm);
let mut th = TopHits::new();

// Search a sequence programmatically (no file I/O needed)
let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
let sq = Sequence {
    name: "my_seq".into(),
    acc: String::new(),
    desc: String::new(),
    dsq,
    n: 20,
    l: 20,
};
pli.run(&gm, &om, &bg, &sq, &mut th);

// Access results
th.sort_by_sortkey();
for hit in &th.hits {
    println!("{}: score={:.1} bits", hit.name, hit.score);
}
```

## Testing

The test suite mixes exact small-fixture parity checks, real-world regression
tests, and broader equivalence sweeps against bundled C outputs.
Source checkouts need the ignored upstream `hmmer/` snapshot prepared locally;
crate packages include only the small upstream tutorial/testsuite data needed
by packaged tests.

```bash
cargo test --release                  # non-ignored tests
cargo test --release -- --ignored     # fixture-heavy tests; requires external data/C HMMER
cargo test --test real_world_regression_tests
cargo test --test pfam_equivalence_tests
cargo test --test jackhmmer_integration_tests
```

Current real-data coverage includes:

- local/generated Pfam golden fixtures against `test_data/human_swissprot_2k.fasta`
- exact and regression-style `jackhmmer` tests on real protein data
- exact `nhmmer` goldens including a no-hit ECORI control
- larger out-of-tree benchmark fixtures documented in `REAL_WORLD_FIXTURES.md`

Benchmark/test policy:

- README performance claims must come from `scripts/benchmark_real_world.sh`
  for search workloads or `scripts/benchmark_utilities.sh` for utility
  workloads.
- Benchmark artifact directories must keep `metadata.txt`, `results.tsv`,
  `run_order.tsv`, command files, git status/diff audit files,
  `.cargo/config.toml` metadata/content or absence, generated-query source
  checksum manifests, and dataset checksum/size manifests with the reported
  numbers.
- Use `ALLOW_MISMATCH=1` only for exploratory timing; do not cite those results
  as parity-checked evidence.
- Fixture-heavy, bundled-C parity, and large Swiss-Prot integration tests are
  excluded from crate packages when their required fixtures or C executables are
  not packaged. Run them from a source checkout after preparing the named
  fixtures and bundled C tools.

## Architecture

- `alphabet` - DNA/RNA/amino acid alphabets with digital encoding
- `hmm` / `hmmfile` - HMM data structures and file I/O
- `bg` - Null/background model with composition bias filter
- `profile` - Scoring profiles and model configuration
- `simd/oprofile` - SSE2-optimized profile (byte, word, and float precision layouts)
- `simd/msv_filter` - SSE2 MSV filter (byte precision, first pipeline stage)
- `simd/vit_filter` - SSE2 Viterbi filter (int16 precision, second stage)
- `simd/fwd_filter` - SSE2 Forward parser (float precision, third stage)
- `dp/` - Generic DP algorithms (Viterbi, Forward, Backward, MSV, Decoding)
- `domaindef` - Domain definition via posterior decoding with btot/etot region detection
- `pipeline` - Multi-stage search pipeline (MSV -> bias filter -> Vit -> Fwd -> domain def)
- `tophits` - Hit collection, sorting, thresholding, and output
- `stats/` - Gumbel and exponential distributions for E-value calculation
- `calibrate` - E-value parameter estimation by simulation

## Remaining Gaps

The command surface is broadly present, but a few areas are still incomplete,
not yet fully C-identical, or still need broader validation:

- `hmmalign` now reconstructs model-guided alignments and supports Stockholm,
  A2M, `--trim`, `-o`, and strict upstream-style `--mapali` checksum
  validation. The remaining gap is breadth, not core functionality: only the
  implemented output formats are accepted, and parity coverage is still growing
  around legacy fixture edge cases.
- `phmmer` and `jackhmmer` now use the upstream-style single-sequence
  score-matrix conversion path instead of the earlier renormalized shortcut.
  Later `jackhmmer` rounds rebuild from model-guided checkpoint alignments,
  with exact bundled-C `--chkali` and `--chkhmm` parity covered on the globins
  fixture. Remaining work is broader iterative-search score parity across more
  real databases and threshold combinations.
- `nhmmer` accepts HMM queries plus nucleotide query models built from
  Stockholm, aligned FASTA, or single-sequence FASTA input. Remaining query
  format aliases beyond `hmm`, `stockholm`/`sto`, `afa`, and `fasta` are
  rejected, and FM-index target databases from `makehmmerdb` are not yet used
  by `nhmmer` search.
- `makehmmerdb` still writes a Rust `HMMERDB\0` container. It carries
  C-layout metadata and FM-record extensions used for convergence testing, but
  the top-level file is not yet the native C FM-index stream and block
  windowing still differs from C `esl_sqio_ReadBlock()`.
- `hmmsim` writes deterministic score/statistical artifacts and supports the
  current calibration-size and Forward tail-mass controls. The remaining gap is
  exact C main-summary table formatting for Viterbi/MSV/hybrid modes.
- `hmmpgmd` supports master/worker flags, legacy line queries, C-style client
  request blocks, search stats, and sequence-level `P7_HIT` shell records. It
  does not yet serialize full domain/alignment payloads or run distributed
  worker-side search/merge.
- `hmmsearch --pfamtblout` now writes both Pfam sections with C-style domain
  ordering and coordinate generation even under `--noali`. Current regressions
  cover exact bundled-C parity on small fixtures plus a real-world GECCO case.
- Performance work is no longer primarily about large-dataset RSS on
  `hmmsearch`; the current remaining performance gap is mostly multi-thread
  scaling and other workload breadth, not the earlier whole-database memory
  retention issue.
- `hmmpress` writes C-compatible `.h3m/.h3f/.h3p/.h3i` sidecars for
  `hmmscan`/`nhmmscan`. Compatibility coverage includes small C-produced
  pressed fixtures and Rust-produced sidecars used by scan parity tests; broader
  database coverage is still growing.

Currently supported programs:
- `hmmsearch` - Search HMM(s) against a sequence database (FASTA/UniProt/gzipped)
- `hmmbuild` - Build profile HMM(s) from Stockholm multiple sequence alignments
- `phmmer` - Search a protein sequence against a protein database
- `jackhmmer` - Iteratively search a protein sequence against a database
- `nhmmer` - Search DNA/RNA HMM(s) against a nucleotide database
- `hmmalign` - Align sequences to a profile HMM
- `hmmstat` - Display summary statistics for each HMM
- `hmmemit` - Emit consensus or sampled sequences from HMM
- `hmmsim` - Simulate score distributions from an HMM
- `hmmconvert` - Convert HMM files (read and rewrite)
- `hmmfetch` - Retrieve HMM from a file by name (with SSI index)
- `hmmlogo` - Generate HMM sequence logo data
- `hmmscan` - Search sequence(s) against a profile HMM database
- `nhmmscan` - Search nucleotide sequence(s) against DNA HMM database
- `hmmpress` - Prepare an HMM database for `hmmscan`/`nhmmscan` by writing
  `.h3m/.h3f/.h3p/.h3i` sidecars
- `alimask` - Add `--alirange` or `--modelmask` mask annotation to Stockholm alignment
- `makehmmerdb` - Create FM-index database for nhmmer
- `hmmpgmd` - Run an HMM or protein sequence database daemon with `--master`,
  `--worker`, `--cport`, `--wport`, legacy line queries, and a compatible subset
  of the C client/worker wire framing

## How to Cite

- HMMER software and documentation: http://hmmer.org/
- Eddy SR. *Accelerated profile HMM searches.* PLoS Comput Biol. 2011;7(10):e1002195.
  doi:[10.1371/journal.pcbi.1002195](https://doi.org/10.1371/journal.pcbi.1002195)

## License

BSD-3-Clause (same as HMMER)
