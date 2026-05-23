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
```

## Datasets

### Protein medium

- Path: `external/protein_medium/uniprot_UP000005640_human.fasta.gz`
- Rewindable path for iterative C `jackhmmer`: `external/protein_medium/uniprot_UP000005640_human.fasta`
- Source page: `https://www.uniprot.org/proteomes/UP000005640`
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
- Intended use:
  - large `phmmer`
  - large `jackhmmer`
  - large `hmmsearch`

### DNA large

- Path: `external/dna_large/ensembl_homo_sapiens_GRCh38_primary_assembly.fa.gz`
- Source page: `https://www.ensembl.org/Homo_sapiens`
- Assembly info: `https://www.ensembl.org/Homo_sapiens/Info/Annotation`
- Intended use:
  - large `nhmmer`

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
artifacts under `reports/benchmarks/<timestamp>/`, and fails if basic output
agreement checks do not pass. The commands below are useful for ad hoc
investigation, but README-visible performance claims should come from the
script output.

Use `CASES=hmmsearch_human_pkinase` for a single workload, or a comma-separated
list of case names for a subset.

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
  external/protein_medium/uniprot_UP000005640_human.fasta.gz
```

### Large `jackhmmer`

```bash
/usr/bin/time -v target/release/hmmer jackhmmer \
  -N 2 --cpu 1 \
  --tblout /tmp/jack.large.tbl \
  --domtblout /tmp/jack.large.domtbl \
  /tmp/query.fa \
  external/protein_large/uniprot_sprot.fasta.gz
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
