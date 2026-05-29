#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rust_bin="${RUST_HMMER:-target/release/hmmer}"
fixture_root="${FIXTURE_ROOT:-external/realistic}"
skip_build="${SKIP_BUILD:-0}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${OUT_DIR:-reports/benchmarks/realistic-all-tools-$stamp}"
failures=0

protein_fasta="$fixture_root/protein/uniprot_UP000002311_yeast.fasta"
protein_fasta_gz="$fixture_root/protein/uniprot_UP000002311_yeast.fasta.gz"
protein_query="$fixture_root/queries/yeast_first_protein.fa"
dna_fasta="$fixture_root/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"
pfam_panel="$(find "$fixture_root/pfam" -maxdepth 1 -name 'Pfam-A.first*.hmm' | sort | tail -n 1)"
pfam_seed="$fixture_root/pfam/PF00226_DnaJ.seed.sto"
rfam_seed_dna="$fixture_root/rfam/RF00005_tRNA.dna.seed.sto"

protein_hmm="$out_dir/derived/PF00226_DnaJ.built.hmm"
protein_hmmdb="$out_dir/derived/Pfam-A.panel.db.hmm"
dna_hmm="$out_dir/derived/RF00005_tRNA.hmm"
dna_hmmdb="$out_dir/derived/RF00005_tRNA.db.hmm"
protein_emit="$out_dir/derived/PF00226_emitted.fa"
fm_index="$out_dir/derived/yeast_genome.hmmerdb"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required fixture: $path" >&2
    echo "prepare fixtures with scripts/download_realistic_fixtures.sh" >&2
    return 1
  fi
}

file_size_bytes() {
  wc -c < "$1" | tr -d ' '
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

first_hmm_name() {
  awk '$1 == "NAME" { print $2; exit }' "$1"
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

  printf "%q " timeout 2s "$rust_bin" pgmd --master --hmmdb "$protein_hmmdb" --cport 51381 --wport 51382 > "$cmd_file"
  echo "running $case_name"
  if /usr/bin/time -v -o "$time_file" timeout 2s "$rust_bin" pgmd --master --hmmdb "$protein_hmmdb" --cport 51381 --wport 51382 > "$stdout" 2> "$stderr"; then
    status=0
  else
    status=$?
  fi

  if [[ "$status" != "0" && "$status" != "124" ]]; then
    failures=$((failures + 1))
  fi
  printf "%s\t%s\t%s\n" "$case_name" "$status" "$cmd_file" >> "$out_dir/status.tsv"
}

if [[ -z "$pfam_panel" ]]; then
  echo "missing required fixture: $fixture_root/pfam/Pfam-A.first*.hmm" >&2
  echo "prepare fixtures with scripts/download_realistic_fixtures.sh" >&2
  exit 1
fi

require_file "$protein_fasta"
require_file "$protein_fasta_gz"
require_file "$protein_query"
require_file "$dna_fasta"
require_file "$pfam_panel"
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

panel_fetch_name="$(first_hmm_name "$pfam_panel")"
if [[ -z "$panel_fetch_name" ]]; then
  echo "could not determine first HMM NAME from $pfam_panel" >&2
  exit 1
fi

printf "case\tstatus\tcommand_file\n" > "$out_dir/status.tsv"
printf "case_dataset\tpath\tstatus\tbytes\tsha256\n" > "$out_dir/datasets.tsv"
record_dataset "yeast_proteome_gz" "$protein_fasta_gz"
record_dataset "yeast_proteome" "$protein_fasta"
record_dataset "yeast_query_first_protein" "$protein_query"
record_dataset "yeast_genome" "$dna_fasta"
record_dataset "pfam_panel" "$pfam_panel"
record_dataset "pfam_PF00226_seed" "$pfam_seed"
record_dataset "rfam_RF00005_dna_seed" "$rfam_seed_dna"
write_metadata

run_tool hmmbuild_pfam_dnaj "$rust_bin" build --amino "$protein_hmm" "$pfam_seed"
run_tool hmmbuild_rfam_trna "$rust_bin" build --dna "$dna_hmm" "$rfam_seed_dna"
cp "$pfam_panel" "$protein_hmmdb"
cp "$dna_hmm" "$dna_hmmdb"

run_tool hmmpress_pfam_panel "$rust_bin" press -f "$protein_hmmdb"
run_tool hmmpress_rfam "$rust_bin" press -f "$dna_hmmdb"
run_tool hmmfetch_pfam_panel "$rust_bin" fetch "$protein_hmmdb" "$panel_fetch_name"
run_tool hmmstat_pfam_panel "$rust_bin" stat "$protein_hmmdb"
run_tool hmmconvert_pfam_dnaj "$rust_bin" convert "$protein_hmm"
run_tool hmmlogo_pfam_dnaj "$rust_bin" logo "$protein_hmm"
run_tool hmmemit_pfam_dnaj "$rust_bin" emit --seed 42 -N 20 "$protein_hmm"
cp "$out_dir/hmmemit_pfam_dnaj.stdout" "$protein_emit"
run_tool hmmalign_pfam_dnaj_emitted "$rust_bin" align "$protein_hmm" "$protein_emit"
run_tool hmmsim_pfam_dnaj "$rust_bin" sim --seed 42 -N 1000 --msv "$protein_hmm"
run_tool alimask_pfam_dnaj "$rust_bin" alimask --amino --modelrange 5..25 "$pfam_seed" "$out_dir/derived/PF00226_masked.sto"
run_tool makehmmerdb_yeast_genome "$rust_bin" makehmmerdb "$dna_fasta" "$fm_index"

run_tool hmmsearch_pfam_panel_yeast "$rust_bin" search --noali --tblout "$out_dir/hmmsearch_pfam_panel_yeast.tblout" "$protein_hmmdb" "$protein_fasta_gz"
run_tool hmmscan_pfam_panel_yeast "$rust_bin" scan --tblout "$out_dir/hmmscan_pfam_panel_yeast.tblout" "$protein_hmmdb" "$protein_fasta_gz"
run_tool phmmer_query_yeast "$rust_bin" phmmer --tblout "$out_dir/phmmer_query_yeast.tblout" "$protein_query" "$protein_fasta_gz"
run_tool jackhmmer_query_yeast "$rust_bin" jackhmmer -N 2 --tblout "$out_dir/jackhmmer_query_yeast.tblout" "$protein_query" "$protein_fasta"
run_tool nhmmer_trna_yeast "$rust_bin" nhmmer --tblout "$out_dir/nhmmer_trna_yeast.tblout" "$dna_hmm" "$dna_fasta"
run_tool nhmmer_trna_yeast_fm "$rust_bin" nhmmer --tblout "$out_dir/nhmmer_trna_yeast_fm.tblout" "$dna_hmm" "$fm_index"
run_tool nhmmscan_trna_yeast "$rust_bin" nhmmscan --tblout "$out_dir/nhmmscan_trna_yeast.tblout" "$dna_hmmdb" "$dna_fasta"
run_pgmd_probe

if [[ "$failures" -ne 0 ]]; then
  echo "$failures realistic all-tools case(s) failed; see $out_dir/status.tsv" >&2
  exit 1
fi

echo "realistic all-tools artifacts written to $out_dir"
