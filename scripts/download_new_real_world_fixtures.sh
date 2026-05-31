#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fixture_root="${FIXTURE_ROOT:-external/new_real}"
refresh="${REFRESH:-0}"

protein_dir="$fixture_root/protein"
dna_dir="$fixture_root/dna"
pfam_dir="$fixture_root/pfam"
rfam_dir="$fixture_root/rfam"
derived_dir="$fixture_root/derived"
query_dir="$fixture_root/queries"

mkdir -p "$protein_dir" "$dna_dir" "$pfam_dir" "$rfam_dir" "$derived_dir" "$query_dir"

uniprot_ecoli_url='https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28proteome%3AUP000000625%29'
ncbi_ecoli_url='https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz'
pfam_seed_url='https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz'
rfam_seed_url='https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz'

download_file() {
  local url="$1"
  local output="$2"
  local tmp="${output}.tmp"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "downloading $url"
  curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"
  mv "$tmp" "$output"
}

decompress_gzip() {
  local input="$1"
  local output="$2"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "decompressing $input"
  gzip -dc "$input" > "$output"
}

extract_stockholm_by_accession() {
  local input_gz="$1"
  local accession="$2"
  local output="$3"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "extracting $accession to $output"
  gzip -dc "$input_gz" | awk -v accession="$accession" '
    BEGIN { RS = "//\n"; ORS = "" }
    !found && $0 ~ "#=GF AC[ \t]+" accession "([.]|[ \t\n])" {
      print $0 "//\n"
      found = 1
    }
    END { if (!found) exit 1 }
  ' > "$output"
}

extract_fasta_by_id() {
  local input="$1"
  local query_id="$2"
  local output="$3"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "extracting $query_id to $output"
  awk -v query_id="$query_id" '
    /^>/ { hit = (index($0, query_id) > 0) }
    hit { print }
  ' "$input" > "$output"
  if [[ ! -s "$output" ]]; then
    echo "could not extract $query_id from $input" >&2
    return 1
  fi
}

write_first_bases() {
  local input="$1"
  local output="$2"
  local limit="$3"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "writing first $limit bases from $input to $output"
  awk -v limit="$limit" '
    /^>/ {
      if (!started) {
        print
        started = 1
      }
      next
    }
    started && bases < limit {
      gsub(/[[:space:]]/, "")
      take = limit - bases
      if (length($0) > take) {
        print substr($0, 1, take)
        bases = limit
      } else {
        print
        bases += length($0)
      }
      if (bases >= limit) exit
    }
  ' "$input" > "$output"
}

sha_manifest() {
  local output="$fixture_root/SHA256SUMS"
  find "$fixture_root" -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output"
}

download_file "$uniprot_ecoli_url" "$protein_dir/uniprot_UP000000625_ecoli_k12.fasta.gz"
download_file "$ncbi_ecoli_url" "$dna_dir/GCF_000005845.2_ASM584v2_genomic.fna.gz"
download_file "$pfam_seed_url" "$pfam_dir/Pfam-A.seed.gz"
download_file "$rfam_seed_url" "$rfam_dir/Rfam.seed.gz"

decompress_gzip "$protein_dir/uniprot_UP000000625_ecoli_k12.fasta.gz" \
  "$protein_dir/uniprot_UP000000625_ecoli_k12.fasta"
decompress_gzip "$dna_dir/GCF_000005845.2_ASM584v2_genomic.fna.gz" \
  "$dna_dir/GCF_000005845.2_ASM584v2_genomic.fna"

extract_stockholm_by_accession "$pfam_dir/Pfam-A.seed.gz" "PF02518" "$pfam_dir/PF02518_HATPase_c.seed.sto"
extract_stockholm_by_accession "$pfam_dir/Pfam-A.seed.gz" "PF00072" "$pfam_dir/PF00072_Response_reg.seed.sto"
extract_stockholm_by_accession "$rfam_dir/Rfam.seed.gz" "RF00005" "$rfam_dir/RF00005_tRNA.seed.sto"
sed 's/U/T/g; s/u/t/g' "$rfam_dir/RF00005_tRNA.seed.sto" > "$rfam_dir/RF00005_tRNA.dna.seed.sto"

extract_fasta_by_id "$protein_dir/uniprot_UP000000625_ecoli_k12.fasta" \
  "sp|P0A6Y8|DNAK_ECOLI" "$query_dir/sp_P0A6Y8_DNAK_ECOLI.fa"
write_first_bases "$dna_dir/GCF_000005845.2_ASM584v2_genomic.fna" \
  "$derived_dir/GCF_000005845.2_ASM584v2_first250k.fna" 250000

{
  echo "fixture_root=$fixture_root"
  echo "uniprot_ecoli_url=$uniprot_ecoli_url"
  echo "ncbi_ecoli_url=$ncbi_ecoli_url"
  echo "pfam_seed_url=$pfam_seed_url"
  echo "rfam_seed_url=$rfam_seed_url"
  echo "pfam_selected_accession=PF02518"
  echo "pfam_extra_accession=PF00072"
  echo "rfam_selected_accession=RF00005"
  echo "protein_query_id=sp|P0A6Y8|DNAK_ECOLI"
  echo "dna_subset_bases=250000"
} > "$fixture_root/sources.tsv"

sha_manifest

cat <<EOF
New real-world fixtures are ready under $fixture_root.

Run all Rust CLI tools on these fixtures with:
  scripts/benchmark_new_real_all_tools.sh
EOF
