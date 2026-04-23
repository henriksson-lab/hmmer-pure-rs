# hmmer-pure-rs 0.7.0

A Rust port of [HMMER 3.4](http://hmmer.org/) for biological sequence analysis using profile hidden Markov models (profile HMMs). Searches sequence databases for homologous sequences.

* 2026-04-22: The code has passed current methods of testing and **careful use on real data is possible**. Up to 4x faster in some cases, which is concerning(!), but obvious reasons for why this might be wrong has at least been checked

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

These benchmarks are here to document the current translation state, not to
promise performance on every machine or workload. All runs below were taken in
the same workspace with `--cpu 1` or `--cpu 4` as shown.

### Medium Fixture

Dataset:

- `external/protein_medium/uniprot_UP000005640_human.fasta(.gz)`
- medium human proteome fixture

Queries:

- `hmmsearch`: `test_data/Pkinase_pfam.hmm`
- `jackhmmer`: `sp|P00738|HPT_HUMAN`

Results:

| Command | Threads | Rust | C | Notes |
|-------|-------:|-------|-------|-------|
| `search --noali` | 1 | `1.42s` user / `1.43s` wall / `16.9 MB` RSS | `6.34s` user / `6.13s` wall / `15.9 MB` RSS | `484` `tblout` rows both |
| `jackhmmer -N 2 --tblout --domtblout` | 1 | `4.61s` user / `5.38s` wall / `18.2 MB` RSS | `8.74s` user / `8.18s` wall / `16.9 MB` RSS | `127` `tblout`, `156` `domtblout` rows both |

### Large Fixture

Dataset:

- `external/protein_large/uniprot_sprot.fasta(.gz)`
- full Swiss-Prot protein set

Queries:

- `hmmsearch`: `test_data/Pkinase_pfam.hmm`
- `jackhmmer`: `sp|P00738|HPT_HUMAN`

Results:

| Command | Threads | Rust | C | Notes |
|-------|-------:|-------|-------|-------|
| `search --noali` | 1 | `13.73s` user / `13.90s` wall / `18.5 MB` RSS | `72.36s` user / `65.17s` wall / `39.4 MB` RSS | `4543` `tblout` rows both |
| `search --noali` | 4 | `21.76s` user / `9.10s` wall / `42.0 MB` RSS | `75.78s` user / `17.72s` wall / `62.3 MB` RSS | `4543` `tblout` rows both |
| `jackhmmer -N 2 --tblout --domtblout` | 1 | `56.16s` user / `57.84s` wall / `29.9 MB` RSS | `97.58s` user / `82.66s` wall / `29.2 MB` RSS | `887` `tblout`, `963` `domtblout` rows both |

Interpretation:

- `hmmsearch` is currently faster than bundled C on the measured medium and
  large fixtures
- `jackhmmer` is also faster than bundled C on the measured medium and large
  fixtures
- the large `hmmsearch` RSS problem that existed earlier is no longer present
  on the measured `--cpu 1` path, and `--cpu 4` remains well below the earlier
  whole-database-retention behavior
- the main remaining performance question is further multi-thread scaling, not
  large single-thread memory usage

## Features

- Pure Rust implementation of the HMMER search pipeline
- SSE2-accelerated MSV, Viterbi, and Forward filters
- Full domain definition with posterior decoding (btot/etot/mocc region detection)
- Stochastic traceback clustering for multi-domain sequences
- Null2 bias correction with omega weighting
- Composition bias filter matching C HMMER
- Reads HMMER3 format HMM files (versions 3a-3f)
- Reads FASTA and gzipped FASTA (.fasta.gz) sequence databases
- All C HMMER hmmsearch flags supported (--cut_ga, -Z, --nobias, --acc, etc.)
- Tabular output (`--tblout`, `--domtblout`, `--pfamtblout`) compatible with HMMER3
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
hmmer scan query.fa hmm_database.hmm

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
```

### hmmsearch flags

```
Output:       -o, --tblout, --domtblout, --pfamtblout, -A, --noali, --acc, --notextw, --textw
Thresholds:   -E, -T, --domE, --domT, --incE, --incT, --incdomE, --incdomT
Cutoffs:      --cut_ga, --cut_nc, --cut_tc
Filters:      --max, --F1, --F2, --F3, --nobias
Expert:       --nonull2, -Z, --domZ, --seed, --tformat, --cpu
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

```bash
cargo test --release                  # all tests
cargo test --test real_world_regression_tests
cargo test --test pfam_equivalence_tests
cargo test --test jackhmmer_integration_tests
```

Current real-data coverage includes:

- committed Pfam golden fixtures against `test_data/human_swissprot_2k.fasta`
- exact and regression-style `jackhmmer` tests on real protein data
- exact `nhmmer` goldens including a no-hit ECORI control
- larger out-of-tree benchmark fixtures documented in `REAL_WORLD_FIXTURES.md`

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
- `hmmsearch --pfamtblout` now writes both Pfam sections with C-style domain
  ordering and coordinate generation even under `--noali`. Current regressions
  cover exact bundled-C parity on small fixtures plus a real-world GECCO case.
- Sequence-level null2 bias is currently covered by exact checked fixtures,
  including a multi-domain `fn3` regression, but the broader validation corpus
  here is still lighter than for the core `hmmsearch`/`nhmmer` search paths.
- Performance work is no longer primarily about large-dataset RSS on
  `hmmsearch`; the current remaining performance gap is mostly multi-thread
  scaling and other workload breadth, not the earlier whole-database memory
  retention issue.

Currently supported programs:
- `hmmsearch` - Search HMM(s) against a sequence database (FASTA/UniProt/gzipped)
- `hmmbuild` - Build profile HMM(s) from Stockholm multiple sequence alignments
- `phmmer` - Search a protein sequence against a protein database
- `jackhmmer` - Iteratively search a protein sequence against a database
- `nhmmer` - Search DNA/RNA HMM(s) against a nucleotide database
- `hmmalign` - Align sequences to a profile HMM
- `hmmstat` - Display summary statistics for each HMM
- `hmmemit` - Emit consensus or sampled sequences from HMM
- `hmmconvert` - Convert HMM files (read and rewrite)
- `hmmfetch` - Retrieve HMM from a file by name (with SSI index)
- `hmmlogo` - Generate HMM sequence logo data
- `hmmscan` - Search sequence(s) against a profile HMM database
- `nhmmscan` - Search nucleotide sequence(s) against DNA HMM database
- `hmmpress` - Prepare HMM database (binary pressed format)
- `alimask` - Add mask annotation to Stockholm alignment
- `makehmmerdb` - Create FM-index database for nhmmer

## License

BSD-3-Clause (same as HMMER)
