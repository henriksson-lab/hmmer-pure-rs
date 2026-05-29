# Real-World Fixtures

This repo keeps large real-world benchmark datasets outside `test_data/` and out
of git. Put them under `external/`.

## Layout

Expected local paths:

```text
external/
  protein_medium/
    uniprot_UP000005640_human.fasta.gz
    uniprot_UP000005640_human.fasta
    queries/
  protein_large/
    uniprot_sprot.fasta.gz
    uniprot_sprot.fasta
    queries/
  dna_large/
    ensembl_homo_sapiens_GRCh38_primary_assembly.fa.gz
    queries/
  new_real/
    protein/
    dna/
    pfam/
    rfam/
    derived/
    queries/
  realistic/
    protein/
    dna/
    pfam/
    rfam/
    queries/
```

## Datasets

### Protein medium

- Path: `external/protein_medium/uniprot_UP000005640_human.fasta.gz`
- Rewindable path for iterative C `jackhmmer`: `external/protein_medium/uniprot_UP000005640_human.fasta`
- Source page: `https://www.uniprot.org/proteomes/UP000005640`
- Source provenance: mutable UniProt proteome download as mirrored on the
  2026-05-28 benchmark host; keep the checksum below with any cited result.
- Local status from the 2026-05-28 benchmark host:
  - gzip bytes: `7750284`
  - gzip SHA-256: `cfaa8ce64eb832a549be794ab86127d49574456708adb756907415949ca2cf58`
  - uncompressed bytes: `13735476`
  - uncompressed SHA-256: `7272526c282498e7229eefedeb34173a52e9d3c19a046102d93c02c72d20dbef`
  - sequences/residues: `20659` / `11456702`
- Intended use:
  - medium `phmmer`
  - medium `jackhmmer`
  - medium `hmmsearch`

### Protein large

- Path: `external/protein_large/uniprot_sprot.fasta.gz`
- Rewindable path for iterative C `jackhmmer`: `external/protein_large/uniprot_sprot.fasta`
- Source page: `https://www.uniprot.org/help/downloads`
- FTP file:
  - `https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz`
- Source provenance: mutable UniProt `current_release` path as mirrored on the
  2026-05-28 benchmark host; keep the checksum below with any cited result.
- Local status from the 2026-05-28 benchmark host:
  - gzip bytes: `93457057`
  - gzip SHA-256: `5ba5cb332fc7794ab1c02075a79c8b3d95b573f9b244a38bb53558172e1f9b7b`
  - uncompressed bytes: `287479954`
  - uncompressed SHA-256: `987e9d468c2691008b8c7d9d3eea79dec207b5c892312f8d63d1fe6f5b01cead`
  - sequences/residues: `574627` / `208482574`
- Intended use:
  - large `phmmer`
  - large `jackhmmer`
  - large `hmmsearch`

### DNA large

- Path: `external/dna_large/ensembl_homo_sapiens_GRCh38_primary_assembly.fa.gz`
- Source page: `https://www.ensembl.org/Homo_sapiens`
- Assembly info: `https://www.ensembl.org/Homo_sapiens/Info/Annotation`
- Local status from the 2026-05-28 benchmark host: missing
- Intended use:
  - future large `nhmmer`
  - current README benchmark nucleotide rows are smoke-sized in-tree cases

## Recommended queries

Reuse existing parity-covered cases where possible.

### `jackhmmer`

- `sp|O43739|CYH3_HUMAN`
- `sp|P00738|HPT_HUMAN`

### `hmmsearch`

- `test_data/Pkinase_pfam.hmm`
- `test_data/Ras_pfam.hmm`

### `nhmmer`

- one existing exact nucleotide golden
- one no-hit control such as the ECORI case

## Benchmark commands

Prefer the checked script over copying individual commands:

```bash
scripts/benchmark_real_world.sh
```

It runs Rust and bundled C HMMER on the same files, records metadata and timing
artifacts under `reports/benchmarks/<timestamp>/`, alternates Rust/C execution
order by round, and fails if basic output agreement checks do not pass. The commands below are useful for ad hoc
investigation, but README-visible performance claims should come from the
script output. The generated `datasets.tsv`/metadata artifacts should be kept
with any cited run so fixture size, checksum, status, command lines, host, git
commit, and dirty-worktree status remain auditable.

Use `CASES=hmmsearch_human_pkinase` for a single workload, or a comma-separated
list of case names for a subset. Unknown selectors and selectors that expand to
zero cases fail before benchmarking. The real-world harness records
`ALLOW_MISMATCH`, `CASES`, `selected_cases`, `run_order.tsv`, git status, git
diff audit checksums, `.cargo/config.toml` content/checksum or absence,
Rust/C executable metadata, generated-query source checksum manifests, dataset
byte counts, dataset SHA-256 checksums, and present/missing fixture status in
its artifacts.

Utility-command performance claims should come from:

```bash
scripts/benchmark_utilities.sh
```

Use `CASES=hmmstat_gecco_cluster1,hmmfetch_gecco_cluster1_valid` for a utility
subset. The utility harness records input byte sizes and SHA-256 checksums,
times Rust and bundled C commands in alternating order by round, records
`run_order.tsv`, `ALLOW_MISMATCH`, `CASES`, git status, git diff audit
checksums, `.cargo/config.toml` content/checksum or absence, and Rust/C
executable metadata, and compares normalized outputs for utility cases where
exact text-output parity is expected. `hmmsim` and `makehmmerdb` are
intentionally timing/status checks only until their known output/container
format gaps are closed.

## New Real-Data All-Tools Smoke Fixtures

To exercise every Rust CLI subcommand on real data separate from the existing
human/Swiss-Prot/GECCO/tutorial fixtures, prepare:

```bash
scripts/download_new_real_world_fixtures.sh
```

The downloader writes under `external/new_real/`, records source URLs in
`external/new_real/sources.tsv`, and records file checksums in
`external/new_real/SHA256SUMS`. The selected fixture sources are:

- UniProt E. coli K-12 proteome `UP000000625`
- NCBI RefSeq E. coli K-12 assembly `GCF_000005845.2_ASM584v2`
- Pfam current-release seed alignment archive, extracting `PF02518`
  (`HATPase_c`)
- Rfam current seed alignment archive, extracting `RF00005` (`tRNA`) and
  deriving a DNA alphabet copy for nucleotide-tool smoke tests

Run all Rust tools against these fixtures with:

```bash
scripts/benchmark_new_real_all_tools.sh
```

The all-tools harness builds protein and nucleotide HMMs from the downloaded
seed alignments, presses/fetches/converts/stats/logos/emits/simulates them,
aligns emitted sequences, masks the real Pfam alignment, builds an FM-index for
the real E. coli DNA subset, runs `hmmsearch`, `hmmscan`, `phmmer`,
`jackhmmer`, `nhmmer`, and `nhmmscan`, and probes `hmmpgmd` startup with a
short timeout. Results and command/status files are written under
`reports/benchmarks/new-real-all-tools-<timestamp>/`.

## Realistic Eukaryotic All-Tools Fixtures

The `new_real` fixtures are a quick smoke tier. For a heavier and more
realistic all-tools run, prepare:

```bash
scripts/download_realistic_fixtures.sh
```

This downloader writes under `external/realistic/`, records sources in
`external/realistic/sources.tsv`, and records checksums in
`external/realistic/SHA256SUMS`. The selected fixture sources are:

- UniProt S. cerevisiae proteome `UP000002311`
- Ensembl Genomes S. cerevisiae `R64-1-1` top-level DNA FASTA
- Pfam current-release `Pfam-A.hmm.gz`, extracting a configurable first-N
  multi-model panel (`PFAM_PANEL_COUNT`, default `12`)
- Pfam current-release seed alignment archive, extracting `PF00226` (`DnaJ`)
- Rfam current seed alignment archive, extracting `RF00005` (`tRNA`) and
  deriving a DNA alphabet copy for nucleotide-tool runs

Run the heavier all-tools workflow with:

```bash
scripts/benchmark_realistic_all_tools.sh
```

The realistic harness builds a fresh `DnaJ` HMM from Pfam seed alignment,
presses a multi-model Pfam panel for `hmmscan`/`hmmfetch`/`hmmpgmd`, searches
the full yeast proteome, runs `phmmer`/`jackhmmer` against the full yeast
proteome, builds an FM-index for the full yeast genome, and runs
`nhmmer`/`nhmmscan` against the full yeast genome. Results are written under
`reports/benchmarks/realistic-all-tools-<timestamp>/`.

Compare speed and maximum RSS against the bundled original C HMMER tools with:

```bash
scripts/benchmark_realistic_compare_c.sh
```

The comparison harness writes paired Rust/C command logs, `/usr/bin/time -v`
files, `results.tsv`, and `summary.tsv` under
`reports/benchmarks/realistic-compare-c-<timestamp>/`. For multi-model
`hmmsearch`, it uses the uncompressed yeast proteome for both implementations
because C HMMER must rewind the target sequence file between query models.

### Medium `phmmer`

```bash
/usr/bin/time -v target/release/hmmer phmmer \
  --cpu 1 \
  /tmp/query.fa \
  external/protein_medium/uniprot_UP000005640_human.fasta.gz
```

### Medium `jackhmmer`

```bash
/usr/bin/time -v target/release/hmmer jackhmmer \
  -N 2 --cpu 1 \
  --tblout /tmp/jack.medium.tbl \
  --domtblout /tmp/jack.medium.domtbl \
  /tmp/query.fa \
  external/protein_medium/uniprot_UP000005640_human.fasta
```

### Large `jackhmmer`

```bash
/usr/bin/time -v target/release/hmmer jackhmmer \
  -N 2 --cpu 1 \
  --tblout /tmp/jack.large.tbl \
  --domtblout /tmp/jack.large.domtbl \
  /tmp/query.fa \
  external/protein_large/uniprot_sprot.fasta
```

### Medium `hmmsearch`

```bash
/usr/bin/time -v target/release/hmmer search \
  --cpu 1 --noali \
  --tblout /tmp/hmmsearch.medium.tbl \
  test_data/Pkinase_pfam.hmm \
  external/protein_medium/uniprot_UP000005640_human.fasta.gz
```

### Large `hmmsearch`

```bash
/usr/bin/time -v target/release/hmmer search \
  --cpu 1 --noali \
  --tblout /tmp/hmmsearch.large.tbl \
  test_data/Pkinase_pfam.hmm \
  external/protein_large/uniprot_sprot.fasta.gz
```

### Large `nhmmer`

```bash
/usr/bin/time -v target/release/hmmer nhmmer \
  --cpu 1 \
  /tmp/query_dna.fa \
  external/dna_large/ensembl_homo_sapiens_GRCh38_primary_assembly.fa.gz
```

## What to record

For each run, record:

- command
- query id
- target dataset
- target byte size and SHA-256 checksum
- fixture status (`present`/`missing`) for expected external files
- `.cargo/config.toml` content and checksum, or explicit absence
- generated query fixture source path and source SHA-256 checksum, when a
  query is extracted from a larger FASTA
- Rust/C executable path, byte size, and SHA-256 checksum
- wall time
- user time
- max RSS
- `tblout` row count
- `domtblout` row count, when applicable
- top hit ordering, when applicable
- Rust commit
- C build mode, if comparing against C

## Notes

- Keep large fixtures out of git.
- Prefer stable official sources and pin the exact local filename once
  downloaded.
- For fair Rust vs C benchmarks, use an explicitly documented C build mode.
