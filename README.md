# hmmer-pure-rs 0.5.0

A Rust port of [HMMER 3.4](http://hmmer.org/) for biological sequence analysis using profile hidden Markov models (profile HMMs). Searches sequence databases for homologous sequences.

This is a translation of the original code and not the authorative implementation. This code should generate bitwise
equal output to the original. Please report any deviations

The aim of this project is to increase performance, especially by providing this code through a type-safe library interface.
The code can also be compiled to be used for webassembly.


## Features

- Pure Rust implementation of the HMMER search pipeline
- SSE2-accelerated MSV filter for fast sequence filtering
- Generic Viterbi and Forward DP algorithms
- Reads HMMER3 format HMM files (versions 3a-3f)
- Reads FASTA format sequence databases
- Tabular output (`--tblout`, `--domtblout`) compatible with HMMER3
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

### Search HMM(s) against a sequence database

```bash
# Basic search
hmmsearch query.hmm sequences.fa

# Save tabular output
hmmsearch --tblout hits.tbl query.hmm sequences.fa

# Save domain tabular output
hmmsearch --domtblout domains.tbl query.hmm sequences.fa

# Skip all filters (slower, but finds weaker hits)
hmmsearch --max query.hmm sequences.fa

# Set E-value threshold
hmmsearch -E 0.001 query.hmm sequences.fa

# Adjust filter thresholds
hmmsearch --F1 0.05 --F2 0.01 --F3 0.001 query.hmm sequences.fa
```

## Library Usage

```rust
use hmmer::{Alphabet, Bg, Pipeline, Profile, OProfile, TopHits};
use hmmer::hmmfile;
use hmmer::profile::{profile_config, reconfig_length, P7_LOCAL};
use hmmer::sequence::Sequence;
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

## Benchmarks

Compiled with `RUSTFLAGS="-C target-cpu=native" cargo build --release` on Linux x86_64.

### Single-threaded
| Test | C HMMER 3.4 | Rust hmmer | Notes |
|------|-------------|------------|-------|
| globins4.hmm vs globins45.fa (45 hits) | 0.054s | 0.212s | Domain def uses generic DP |
| minipfam.hmm vs globins45.fa (no hits) | 0.048s | 0.030s | **1.6x faster** |

### Multi-threaded
| Test | C (--cpu 2) | Rust (--cpu 4) | Notes |
|------|-------------|----------------|-------|
| globins4.hmm vs globins45.fa (45 hits) | 0.056s | 0.068s | Near parity with rayon |

The Rust version uses a full SSE2-accelerated pipeline for filtering: MSV (byte), Viterbi (int16), Forward (float). When filters reject sequences (typical real-world), Rust is 1.6x faster. For hit-rich searches, rayon multi-threading brings performance within 20% of C. The Rust version finds the same number of hits as the C version.

## Architecture

- `alphabet` - DNA/RNA/amino acid alphabets with digital encoding
- `hmm` / `hmmfile` - HMM data structures and file I/O
- `bg` - Null/background model
- `profile` - Scoring profiles and model configuration
- `simd/oprofile` - SSE2-optimized profile (byte, word, and float precision layouts)
- `simd/msv_filter` - SSE2 MSV filter (byte precision, first pipeline stage)
- `simd/vit_filter` - SSE2 Viterbi filter (int16 precision, second stage)
- `simd/fwd_filter` - SSE2 Forward parser (float precision, third stage)
- `dp/` - Generic DP algorithms (Viterbi, Forward, Backward, MSV, Decoding)
- `domaindef` - Domain definition via posterior decoding
- `pipeline` - Multi-stage search pipeline (SIMD MSV -> SIMD Vit -> SIMD Fwd -> domain def)
- `tophits` - Hit collection, sorting, and output
- `stats/` - Gumbel and exponential distributions for E-value calculation

## Status

This is an active port of HMMER 3.4 (64 Rust files, 12,600+ lines, 16 programs).

Currently supported programs:
- `hmmsearch` - Search HMM(s) against a sequence database (FASTA/UniProt)
- `hmmbuild` - Build profile HMM(s) from Stockholm multiple sequence alignments
- `phmmer` - Search a protein sequence against a protein database
- `jackhmmer` - Iteratively search a protein sequence against a database
- `nhmmer` - Search DNA/RNA HMM(s) against a nucleotide database (basic, no FM-index)
- `hmmalign` - Align sequences to a profile HMM (simplified)
- `hmmstat` - Display summary statistics for each HMM
- `hmmemit` - Emit consensus or sampled sequences from HMM
- `hmmconvert` - Convert HMM files (read and rewrite)
- `hmmfetch` - Retrieve HMM from a file by name (with SSI index)
- `hmmlogo` - Generate HMM sequence logo data
- `nhmmscan` - Search nucleotide sequence(s) against DNA HMM database
- `alimask` - Add mask annotation to Stockholm alignment
- `makehmmerdb` - Create FM-index database for nhmmer
- `hmmscan` - Search sequence(s) against a profile HMM database
- `hmmpress` - Prepare HMM database (binary pressed format)

Features implemented:
- Full SSE2 SIMD pipeline (MSV + Viterbi + Forward)
- Domain definition with posterior decoding and Viterbi traceback
- Text alignment display (model/match/target lines)
- Multi-threading with rayon (`--cpu N`)
- Tabular output (--tblout, --domtblout)
- E-value calibration by simulation
- Null2 bias correction
- Stockholm MSA reading
- FASTA and UniProt/SwissProt sequence format support
- Pure Rust build (no C compiler needed with `--no-default-features`)

Future optimization:
- AVX2/NEON Viterbi and Forward filters (MSV filters done for both)
- Effective sequence number estimation (entropy-based) for hmmbuild

## License

BSD-3-Clause (same as HMMER)
