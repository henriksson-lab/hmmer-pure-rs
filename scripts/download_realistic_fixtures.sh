#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fixture_root="${FIXTURE_ROOT:-external/realistic}"
refresh="${REFRESH:-0}"
pfam_panel_count="${PFAM_PANEL_COUNT:-12}"

protein_dir="$fixture_root/protein"
dna_dir="$fixture_root/dna"
pfam_dir="$fixture_root/pfam"
rfam_dir="$fixture_root/rfam"
query_dir="$fixture_root/queries"

mkdir -p "$protein_dir" "$dna_dir" "$pfam_dir" "$rfam_dir" "$query_dir"

uniprot_yeast_url='https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28proteome%3AUP000002311%29'
ensembl_yeast_genome_url='https://ftp.ensemblgenomes.ebi.ac.uk/pub/fungi/current/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz'
pfam_hmm_url='https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz'
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

local_or_download() {
  local fallback="$1"
  local url="$2"
  local output="$3"

  if [[ -s "$fallback" && "$refresh" != "1" ]]; then
    echo "using local source $fallback" >&2
    printf "%s\n" "$fallback"
  else
    download_file "$url" "$output" >&2
    printf "%s\n" "$output"
  fi
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

extract_hmm_panel() {
  local input_gz="$1"
  local output="$2"
  local count="$3"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "extracting first $count Pfam HMMs to $output"
  gzip -dc "$input_gz" | awk -v max_records="$count" '
    BEGIN { RS = "//\n"; ORS = "" }
    NF && records < max_records {
      print $0 "//\n"
      records++
    }
    END { if (records < max_records) exit 1 }
  ' > "$output"
}

extract_first_fasta_record() {
  local input="$1"
  local output="$2"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi

  echo "extracting first FASTA record from $input to $output"
  awk '
    /^>/ {
      if (seen) exit
      seen = 1
    }
    seen { print }
    END { if (!seen) exit 1 }
  ' "$input" > "$output"
}

first_hmm_name() {
  awk '$1 == "NAME" { print $2; exit }' "$1"
}

sha_manifest() {
  local output="$fixture_root/SHA256SUMS"
  find "$fixture_root" -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$output"
}

download_file "$uniprot_yeast_url" "$protein_dir/uniprot_UP000002311_yeast.fasta.gz"
download_file "$ensembl_yeast_genome_url" "$dna_dir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz"
download_file "$pfam_hmm_url" "$pfam_dir/Pfam-A.hmm.gz"

pfam_seed_source="$(local_or_download external/new_real/pfam/Pfam-A.seed.gz "$pfam_seed_url" "$pfam_dir/Pfam-A.seed.gz")"
rfam_seed_source="$(local_or_download external/new_real/rfam/Rfam.seed.gz "$rfam_seed_url" "$rfam_dir/Rfam.seed.gz")"

decompress_gzip "$protein_dir/uniprot_UP000002311_yeast.fasta.gz" \
  "$protein_dir/uniprot_UP000002311_yeast.fasta"
decompress_gzip "$dna_dir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz" \
  "$dna_dir/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"

extract_hmm_panel "$pfam_dir/Pfam-A.hmm.gz" "$pfam_dir/Pfam-A.first${pfam_panel_count}.hmm" "$pfam_panel_count"
extract_stockholm_by_accession "$pfam_seed_source" "PF00226" "$pfam_dir/PF00226_DnaJ.seed.sto"
extract_stockholm_by_accession "$rfam_seed_source" "RF00005" "$rfam_dir/RF00005_tRNA.seed.sto"
sed 's/U/T/g; s/u/t/g' "$rfam_dir/RF00005_tRNA.seed.sto" > "$rfam_dir/RF00005_tRNA.dna.seed.sto"
extract_first_fasta_record "$protein_dir/uniprot_UP000002311_yeast.fasta" "$query_dir/yeast_first_protein.fa"

panel_first_name="$(first_hmm_name "$pfam_dir/Pfam-A.first${pfam_panel_count}.hmm")"
{
  echo "fixture_root=$fixture_root"
  echo "uniprot_yeast_url=$uniprot_yeast_url"
  echo "ensembl_yeast_genome_url=$ensembl_yeast_genome_url"
  echo "pfam_hmm_url=$pfam_hmm_url"
  echo "pfam_seed_url=$pfam_seed_url"
  echo "pfam_seed_source=$pfam_seed_source"
  echo "rfam_seed_url=$rfam_seed_url"
  echo "rfam_seed_source=$rfam_seed_source"
  echo "pfam_panel_count=$pfam_panel_count"
  echo "pfam_panel_first_name=$panel_first_name"
  echo "pfam_build_seed_accession=PF00226"
  echo "rfam_selected_accession=RF00005"
  echo "protein_query=first FASTA record from yeast proteome"
} > "$fixture_root/sources.tsv"

sha_manifest

cat <<EOF
Realistic fixtures are ready under $fixture_root.

Run the heavier all-tools workflow with:
  scripts/benchmark_realistic_all_tools.sh
EOF
