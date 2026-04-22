#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
  "$repo_root/external/protein_medium/queries" \
  "$repo_root/external/protein_large/queries" \
  "$repo_root/external/dna_large/queries"

cat <<'EOF'
Created fixture directories:
  external/protein_medium/queries
  external/protein_large/queries
  external/dna_large/queries

Expected dataset filenames:
  external/protein_medium/uniprot_UP000005640_human.fasta.gz
  external/protein_large/uniprot_sprot.fasta.gz
  external/dna_large/ensembl_homo_sapiens_GRCh38_primary_assembly.fa.gz

See REAL_WORLD_FIXTURES.md for sources and benchmark commands.
EOF
