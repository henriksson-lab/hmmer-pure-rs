#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rust_bin="${RUST_HMMER:-target/release/hmmer}"
fixture_root="${FIXTURE_ROOT:-external/new_real}"
skip_build="${SKIP_BUILD:-0}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${OUT_DIR:-reports/benchmarks/new-real-all-tools-$stamp}"
failures=0

protein_fasta="$fixture_root/protein/uniprot_UP000000625_ecoli_k12.fasta"
protein_fasta_gz="$fixture_root/protein/uniprot_UP000000625_ecoli_k12.fasta.gz"
protein_query="$fixture_root/queries/sp_P0A6Y8_DNAK_ECOLI.fa"
dna_fasta="$fixture_root/dna/GCF_000005845.2_ASM584v2_genomic.fna"
dna_subset="$fixture_root/derived/GCF_000005845.2_ASM584v2_first250k.fna"
pfam_seed="$fixture_root/pfam/PF02518_HATPase_c.seed.sto"
rfam_seed_dna="$fixture_root/rfam/RF00005_tRNA.dna.seed.sto"

protein_hmm="$out_dir/derived/PF02518_HATPase_c.hmm"
protein_hmmdb="$out_dir/derived/PF02518_HATPase_c.db.hmm"
dna_hmm="$out_dir/derived/RF00005_tRNA.hmm"
dna_hmmdb="$out_dir/derived/RF00005_tRNA.db.hmm"
protein_emit="$out_dir/derived/PF02518_emitted.fa"
fm_index="$out_dir/derived/ecoli_first250k.hmmerdb"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required fixture: $path" >&2
    echo "prepare fixtures with scripts/download_new_real_world_fixtures.sh" >&2
    return 1
  fi
}

file_size_bytes() {
  wc -c < "$1" | tr -d ' '
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

append_path_metadata() {
  local label="$1"
  local path="$2"

  echo "${label}_path=$path"
  if [[ -e "$path" ]]; then
    echo "${label}_status=present"
    if [[ -f "$path" ]]; then
      echo "${label}_bytes=$(file_size_bytes "$path")"
      echo "${label}_sha256=$(file_sha256 "$path")"
    fi
  else
    echo "${label}_status=missing"
  fi
}

record_dataset() {
  local label="$1"
  local path="$2"
  local status="missing"
  local bytes=""
  local sha=""

  if [[ -f "$path" ]]; then
    status="present"
    bytes="$(file_size_bytes "$path")"
    sha="$(file_sha256 "$path")"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "$label" "$path" "$status" "$bytes" "$sha" >> "$out_dir/datasets.tsv"
}

write_metadata() {
  local status_file="$out_dir/git-status-short.txt"
  local diff_stat_file="$out_dir/git-diff-stat.txt"
  local diff_file="$out_dir/git-diff.patch"

  git status --short > "$status_file" 2>/dev/null || true
  git diff --stat HEAD > "$diff_stat_file" 2>/dev/null || true
  git diff HEAD > "$diff_file" 2>/dev/null || true

  {
    echo "timestamp_utc=$stamp"
    echo "repo_root=$repo_root"
    echo "fixture_root=$fixture_root"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_file=$status_file"
    echo "git_status_short_sha256=$(file_sha256 "$status_file")"
    echo "git_diff_stat_file=$diff_stat_file"
    echo "git_diff_stat_sha256=$(file_sha256 "$diff_stat_file")"
    echo "git_diff_file=$diff_file"
    echo "git_diff_sha256=$(file_sha256 "$diff_file")"
    echo "cargo=$(cargo --version 2>/dev/null || true)"
    echo "rustc=$(rustc --version 2>/dev/null || true)"
    echo "rust_bin=$rust_bin"
    append_path_metadata "rust_bin" "$rust_bin"
    append_path_metadata "fixture_sources" "$fixture_root/sources.tsv"
    append_path_metadata "fixture_sha256sums" "$fixture_root/SHA256SUMS"
    echo "skip_build=$skip_build"
    echo "uname=$(uname -a)"
  } > "$out_dir/metadata.txt"
}

run_tool() {
  local case_name="$1"
  shift
  local stdout="$out_dir/${case_name}.stdout"
  local stderr="$out_dir/${case_name}.stderr"
  local time_file="$out_dir/${case_name}.time"
  local cmd_file="$out_dir/${case_name}.cmd"
  local status=0

  printf "%q " "$@" > "$cmd_file"
  echo "running $case_name"
  if /usr/bin/time -v -o "$time_file" "$@" > "$stdout" 2> "$stderr"; then
    status=0
  else
    status=$?
    failures=$((failures + 1))
  fi

  printf "%s\t%s\t%s\n" "$case_name" "$status" "$cmd_file" >> "$out_dir/status.tsv"
}

run_pgmd_probe() {
  local case_name="hmmpgmd_startup"
  local stdout="$out_dir/${case_name}.stdout"
  local stderr="$out_dir/${case_name}.stderr"
  local time_file="$out_dir/${case_name}.time"
  local cmd_file="$out_dir/${case_name}.cmd"
  local status=0

  printf "%q " timeout 2s "$rust_bin" pgmd --master --hmmdb "$protein_hmmdb" --cport 51379 --wport 51380 > "$cmd_file"
  echo "running $case_name"
  if /usr/bin/time -v -o "$time_file" timeout 2s "$rust_bin" pgmd --master --hmmdb "$protein_hmmdb" --cport 51379 --wport 51380 > "$stdout" 2> "$stderr"; then
    status=0
  else
    status=$?
  fi

  if [[ "$status" != "0" && "$status" != "124" ]]; then
    failures=$((failures + 1))
  fi
  printf "%s\t%s\t%s\n" "$case_name" "$status" "$cmd_file" >> "$out_dir/status.tsv"
}

require_file "$protein_fasta"
require_file "$protein_fasta_gz"
require_file "$protein_query"
require_file "$dna_fasta"
require_file "$dna_subset"
require_file "$pfam_seed"
require_file "$rfam_seed_dna"

mkdir -p "$out_dir/derived"
if [[ "$skip_build" != "1" ]]; then
  cargo build --release
fi
if [[ ! -x "$rust_bin" ]]; then
  echo "missing executable: $rust_bin" >&2
  exit 1
fi

printf "case\tstatus\tcommand_file\n" > "$out_dir/status.tsv"
printf "case_dataset\tpath\tstatus\tbytes\tsha256\n" > "$out_dir/datasets.tsv"
record_dataset "ecoli_k12_proteome_gz" "$protein_fasta_gz"
record_dataset "ecoli_k12_proteome" "$protein_fasta"
record_dataset "ecoli_dnak_query" "$protein_query"
record_dataset "ecoli_k12_genome" "$dna_fasta"
record_dataset "ecoli_k12_genome_first250k" "$dna_subset"
record_dataset "pfam_PF02518_seed" "$pfam_seed"
record_dataset "rfam_RF00005_dna_seed" "$rfam_seed_dna"
write_metadata

run_tool hmmbuild_pfam "$rust_bin" build --amino "$protein_hmm" "$pfam_seed"
run_tool hmmbuild_rfam "$rust_bin" build --dna "$dna_hmm" "$rfam_seed_dna"
cp "$protein_hmm" "$protein_hmmdb"
cp "$dna_hmm" "$dna_hmmdb"

run_tool hmmpress_pfam "$rust_bin" press -f "$protein_hmmdb"
run_tool hmmpress_rfam "$rust_bin" press -f "$dna_hmmdb"
run_tool hmmfetch_pfam "$rust_bin" fetch "$protein_hmmdb" HATPase_c
run_tool hmmstat_pfam "$rust_bin" stat "$protein_hmm"
run_tool hmmconvert_pfam "$rust_bin" convert "$protein_hmm"
run_tool hmmlogo_pfam "$rust_bin" logo "$protein_hmm"
run_tool hmmemit_pfam "$rust_bin" emit --seed 42 -N 5 "$protein_hmm"
cp "$out_dir/hmmemit_pfam.stdout" "$protein_emit"
run_tool hmmalign_pfam_emitted "$rust_bin" align "$protein_hmm" "$protein_emit"
run_tool hmmsim_pfam "$rust_bin" sim --seed 42 -N 100 --msv "$protein_hmm"
run_tool alimask_pfam "$rust_bin" alimask --amino --modelrange 2..10 "$pfam_seed" "$out_dir/derived/PF02518_masked.sto"
run_tool makehmmerdb_ecoli_subset "$rust_bin" makehmmerdb "$dna_subset" "$fm_index"

run_tool hmmsearch_pfam_ecoli "$rust_bin" search --noali --tblout "$out_dir/hmmsearch_pfam_ecoli.tblout" "$protein_hmm" "$protein_fasta_gz"
run_tool hmmscan_pfam_ecoli "$rust_bin" scan --tblout "$out_dir/hmmscan_pfam_ecoli.tblout" "$protein_hmmdb" "$protein_fasta_gz"
run_tool phmmer_dnak_ecoli "$rust_bin" phmmer --tblout "$out_dir/phmmer_dnak_ecoli.tblout" "$protein_query" "$protein_fasta_gz"
run_tool jackhmmer_dnak_ecoli "$rust_bin" jackhmmer -N 2 --tblout "$out_dir/jackhmmer_dnak_ecoli.tblout" "$protein_query" "$protein_fasta"
run_tool nhmmer_trna_ecoli "$rust_bin" nhmmer --tblout "$out_dir/nhmmer_trna_ecoli.tblout" "$dna_hmm" "$dna_subset"
run_tool nhmmer_trna_ecoli_fm "$rust_bin" nhmmer --tblout "$out_dir/nhmmer_trna_ecoli_fm.tblout" "$dna_hmm" "$fm_index"
run_tool nhmmscan_trna_ecoli "$rust_bin" nhmmscan --tblout "$out_dir/nhmmscan_trna_ecoli.tblout" "$dna_hmmdb" "$dna_subset"
run_pgmd_probe

if [[ "$failures" -ne 0 ]]; then
  echo "$failures new-real all-tools case(s) failed; see $out_dir/status.tsv" >&2
  exit 1
fi

echo "new-real all-tools artifacts written to $out_dir"
