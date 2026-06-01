#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rust_bin="${RUST_HMMER:-target/release/hmmer}"
c_dir="${C_HMMER_DIR:-hmmer/src}"
fixture_root="${FIXTURE_ROOT:-external/realistic}"
skip_build="${SKIP_BUILD:-0}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${OUT_DIR:-reports/benchmarks/realistic-compare-c-$stamp}"
failures=0

protein_fasta="$fixture_root/protein/uniprot_UP000002311_yeast.fasta"
protein_fasta_gz="$fixture_root/protein/uniprot_UP000002311_yeast.fasta.gz"
protein_query="$fixture_root/queries/yeast_first_protein.fa"
dna_fasta="$fixture_root/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"
pfam_panel="$(find "$fixture_root/pfam" -maxdepth 1 -name 'Pfam-A.first*.hmm' | sort | tail -n 1)"
pfam_seed="$fixture_root/pfam/PF00226_DnaJ.seed.sto"
rfam_seed_dna="$fixture_root/rfam/RF00005_tRNA.dna.seed.sto"

protein_hmm="$out_dir/derived/PF00226_DnaJ.prepared.hmm"
dna_hmm="$out_dir/derived/RF00005_tRNA.prepared.hmm"
protein_emit="$out_dir/derived/PF00226_emitted.fa"
rust_panel_db="$out_dir/derived/rust.Pfam-A.panel.db.hmm"
c_panel_db="$out_dir/derived/c.Pfam-A.panel.db.hmm"
rust_dna_db="$out_dir/derived/rust.RF00005_tRNA.db.hmm"
c_dna_db="$out_dir/derived/c.RF00005_tRNA.db.hmm"
rust_fm_index="$out_dir/derived/rust.yeast_genome.hmmerdb"
c_fm_index="$out_dir/derived/c.yeast_genome.hmmerdb"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required fixture: $path" >&2
    return 1
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "missing executable: $path" >&2
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

parse_time_field() {
  local path="$1"
  local key="$2"
  awk -v key="$key" '
    BEGIN { prefix = key ":" }
    {
      line = $0
      sub(/^[ \t]+/, "", line)
    }
    index(line, prefix) == 1 {
      value = substr(line, length(prefix) + 1)
      sub(/^[ \t]+/, "", value)
      print value
    }
  ' "$path" | tail -n 1
}

elapsed_to_seconds() {
  local elapsed="$1"
  awk -v t="$elapsed" '
    BEGIN {
      n = split(t, a, ":")
      if (n == 1) print a[1] + 0
      else if (n == 2) print (a[1] * 60) + a[2]
      else print (a[1] * 3600) + (a[2] * 60) + a[3]
    }
  '
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
    echo "git_status_short_sha256=$(file_sha256 "$status_file")"
    echo "git_diff_stat_sha256=$(file_sha256 "$diff_stat_file")"
    echo "git_diff_sha256=$(file_sha256 "$diff_file")"
    echo "rust_bin=$rust_bin"
    echo "rust_bin_bytes=$(file_size_bytes "$rust_bin")"
    echo "rust_bin_sha256=$(file_sha256 "$rust_bin")"
    echo "c_hmmer_dir=$c_dir"
    echo "c_hmmsearch_bytes=$(file_size_bytes "$c_dir/hmmsearch")"
    echo "c_hmmsearch_sha256=$(file_sha256 "$c_dir/hmmsearch")"
    echo "cargo=$(cargo --version 2>/dev/null || true)"
    echo "rustc=$(rustc --version 2>/dev/null || true)"
    echo "skip_build=$skip_build"
    echo "uname=$(uname -a)"
  } > "$out_dir/metadata.txt"
}

append_result() {
  local case_name="$1"
  local impl="$2"
  local status="$3"
  local cmd_file="$4"
  local time_file="$5"
  local elapsed user system rss wall_seconds

  elapsed="$(parse_time_field "$time_file" "Elapsed (wall clock) time (h:mm:ss or m:ss)")"
  user="$(parse_time_field "$time_file" "User time (seconds)")"
  system="$(parse_time_field "$time_file" "System time (seconds)")"
  rss="$(parse_time_field "$time_file" "Maximum resident set size (kbytes)")"
  wall_seconds="$(elapsed_to_seconds "$elapsed")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$case_name" "$impl" "$status" "$wall_seconds" "$user" "$system" "$rss" "$cmd_file" >> "$out_dir/results.tsv"
}

run_one() {
  local case_name="$1"
  local impl="$2"
  shift 2
  local prefix="$out_dir/${case_name}.${impl}"
  local cmd_file="${prefix}.cmd"
  local stdout="${prefix}.stdout"
  local stderr="${prefix}.stderr"
  local time_file="${prefix}.time"
  local status=0

  printf "%q " "$@" > "$cmd_file"
  echo "running $case_name: $impl"
  if /usr/bin/time -v -o "$time_file" "$@" > "$stdout" 2> "$stderr"; then
    status=0
  else
    status=$?
    failures=$((failures + 1))
  fi
  append_result "$case_name" "$impl" "$status" "$cmd_file" "$time_file"
}

run_pair() {
  local case_name="$1"
  shift
  local rust_count="$1"
  shift
  local rust_args=()
  local c_args=()
  local i

  for ((i = 0; i < rust_count; i++)); do
    rust_args+=("$1")
    shift
  done
  c_args=("$@")

  run_one "$case_name" rust "${rust_args[@]}"
  run_one "$case_name" c "${c_args[@]}"
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
require_executable "$c_dir/hmmbuild"
require_executable "$c_dir/hmmpress"
require_executable "$c_dir/hmmfetch"
require_executable "$c_dir/hmmstat"
require_executable "$c_dir/hmmconvert"
require_executable "$c_dir/hmmlogo"
require_executable "$c_dir/hmmemit"
require_executable "$c_dir/hmmalign"
require_executable "$c_dir/hmmsim"
require_executable "$c_dir/alimask"
require_executable "$c_dir/makehmmerdb"
require_executable "$c_dir/hmmsearch"
require_executable "$c_dir/hmmscan"
require_executable "$c_dir/phmmer"
require_executable "$c_dir/jackhmmer"
require_executable "$c_dir/nhmmer"
require_executable "$c_dir/nhmmscan"

mkdir -p "$out_dir/derived"
if [[ "$skip_build" != "1" ]]; then
  cargo build --release
fi
require_executable "$rust_bin"

panel_fetch_name="$(first_hmm_name "$pfam_panel")"
if [[ -z "$panel_fetch_name" ]]; then
  echo "could not determine first HMM NAME from $pfam_panel" >&2
  exit 1
fi

printf "case\timpl\tstatus\twall_seconds\tuser_seconds\tsystem_seconds\tmax_rss_kb\tcommand_file\n" > "$out_dir/results.tsv"
printf "case_dataset\tpath\tstatus\tbytes\tsha256\n" > "$out_dir/datasets.tsv"
record_dataset "yeast_proteome_gz" "$protein_fasta_gz"
record_dataset "yeast_proteome" "$protein_fasta"
record_dataset "yeast_query_first_protein" "$protein_query"
record_dataset "yeast_genome" "$dna_fasta"
record_dataset "pfam_panel" "$pfam_panel"
record_dataset "pfam_PF00226_seed" "$pfam_seed"
record_dataset "rfam_RF00005_dna_seed" "$rfam_seed_dna"
write_metadata

# Shared prepared inputs for search/scan cases.
"$rust_bin" build --amino "$protein_hmm" "$pfam_seed" > "$out_dir/prepare.hmmbuild_pfam.stdout" 2> "$out_dir/prepare.hmmbuild_pfam.stderr"
"$rust_bin" build --dna "$dna_hmm" "$rfam_seed_dna" > "$out_dir/prepare.hmmbuild_rfam.stdout" 2> "$out_dir/prepare.hmmbuild_rfam.stderr"
"$rust_bin" emit --seed 42 -N 20 "$protein_hmm" > "$protein_emit" 2> "$out_dir/prepare.hmmemit.stderr"
cp "$pfam_panel" "$rust_panel_db"
cp "$pfam_panel" "$c_panel_db"
cp "$dna_hmm" "$rust_dna_db"
cp "$dna_hmm" "$c_dna_db"

run_pair hmmbuild_pfam_dnaj 5 "$rust_bin" build --amino "$out_dir/derived/rust.PF00226_DnaJ.hmm" "$pfam_seed" \
  "$c_dir/hmmbuild" "$out_dir/derived/c.PF00226_DnaJ.hmm" "$pfam_seed"
run_pair hmmbuild_rfam_trna 5 "$rust_bin" build --dna "$out_dir/derived/rust.RF00005_tRNA.hmm" "$rfam_seed_dna" \
  "$c_dir/hmmbuild" --dna "$out_dir/derived/c.RF00005_tRNA.hmm" "$rfam_seed_dna"

run_pair hmmpress_pfam_panel 4 "$rust_bin" press -f "$rust_panel_db" \
  "$c_dir/hmmpress" -f "$c_panel_db"
run_pair hmmpress_rfam 4 "$rust_bin" press -f "$rust_dna_db" \
  "$c_dir/hmmpress" -f "$c_dna_db"

run_pair hmmfetch_pfam_panel 4 "$rust_bin" fetch "$rust_panel_db" "$panel_fetch_name" \
  "$c_dir/hmmfetch" "$c_panel_db" "$panel_fetch_name"
run_pair hmmstat_pfam_panel 3 "$rust_bin" stat "$rust_panel_db" \
  "$c_dir/hmmstat" "$c_panel_db"
run_pair hmmconvert_pfam_dnaj 3 "$rust_bin" convert "$protein_hmm" \
  "$c_dir/hmmconvert" "$protein_hmm"
run_pair hmmlogo_pfam_dnaj 3 "$rust_bin" logo "$protein_hmm" \
  "$c_dir/hmmlogo" "$protein_hmm"
run_pair hmmemit_pfam_dnaj 7 "$rust_bin" emit --seed 42 -N 20 "$protein_hmm" \
  "$c_dir/hmmemit" --seed 42 -N 20 "$protein_hmm"
run_pair hmmalign_pfam_dnaj_emitted 4 "$rust_bin" align "$protein_hmm" "$protein_emit" \
  "$c_dir/hmmalign" "$protein_hmm" "$protein_emit"
run_pair hmmsim_pfam_dnaj 8 "$rust_bin" sim --seed 42 -N 1000 --msv "$protein_hmm" \
  "$c_dir/hmmsim" --seed 42 -N 1000 --msv "$protein_hmm"
run_pair alimask_pfam_dnaj 7 "$rust_bin" alimask --amino --modelrange 5..25 "$pfam_seed" "$out_dir/derived/rust.PF00226_masked.sto" \
  "$c_dir/alimask" --amino --modelrange 5..25 "$pfam_seed" "$out_dir/derived/c.PF00226_masked.sto"
run_pair makehmmerdb_yeast_genome 4 "$rust_bin" makehmmerdb "$dna_fasta" "$rust_fm_index" \
  "$c_dir/makehmmerdb" "$dna_fasta" "$c_fm_index"

run_pair hmmsearch_pfam_panel_yeast 7 "$rust_bin" search --tblout "$out_dir/hmmsearch_pfam_panel_yeast.rust.tblout" --noali "$rust_panel_db" "$protein_fasta" \
  "$c_dir/hmmsearch" --noali --tblout "$out_dir/hmmsearch_pfam_panel_yeast.c.tblout" "$c_panel_db" "$protein_fasta"
run_pair hmmscan_pfam_panel_yeast 6 "$rust_bin" scan --tblout "$out_dir/hmmscan_pfam_panel_yeast.rust.tblout" "$rust_panel_db" "$protein_fasta_gz" \
  "$c_dir/hmmscan" --tblout "$out_dir/hmmscan_pfam_panel_yeast.c.tblout" "$c_panel_db" "$protein_fasta_gz"
run_pair phmmer_query_yeast 6 "$rust_bin" phmmer --tblout "$out_dir/phmmer_query_yeast.rust.tblout" "$protein_query" "$protein_fasta_gz" \
  "$c_dir/phmmer" --tblout "$out_dir/phmmer_query_yeast.c.tblout" "$protein_query" "$protein_fasta_gz"
run_pair jackhmmer_query_yeast 8 "$rust_bin" jackhmmer -N 2 --tblout "$out_dir/jackhmmer_query_yeast.rust.tblout" "$protein_query" "$protein_fasta" \
  "$c_dir/jackhmmer" -N 2 --tblout "$out_dir/jackhmmer_query_yeast.c.tblout" "$protein_query" "$protein_fasta"
run_pair nhmmer_trna_yeast 6 "$rust_bin" nhmmer --tblout "$out_dir/nhmmer_trna_yeast.rust.tblout" "$dna_hmm" "$dna_fasta" \
  "$c_dir/nhmmer" --tblout "$out_dir/nhmmer_trna_yeast.c.tblout" "$dna_hmm" "$dna_fasta"
run_pair nhmmer_trna_yeast_fm 6 "$rust_bin" nhmmer --tblout "$out_dir/nhmmer_trna_yeast_fm.rust.tblout" "$dna_hmm" "$rust_fm_index" \
  "$c_dir/nhmmer" --tblout "$out_dir/nhmmer_trna_yeast_fm.c.tblout" "$dna_hmm" "$c_fm_index"
run_pair nhmmscan_trna_yeast 6 "$rust_bin" nhmmscan --tblout "$out_dir/nhmmscan_trna_yeast.rust.tblout" "$rust_dna_db" "$dna_fasta" \
  "$c_dir/nhmmscan" --tblout "$out_dir/nhmmscan_trna_yeast.c.tblout" "$c_dna_db" "$dna_fasta"

normalize_tblout_for_parity() {
  local input="$1"
  local output="$2"
  sed -E \
    -e 's#hmmer/src/(hmmsearch|hmmscan|phmmer|jackhmmer|nhmmer|nhmmscan)#\1#g' \
    -e 's|^(# Date:            ).*|\1<normalized>|' \
    -e 's#\.rust\.tblout#\.tblout#g' \
    -e 's#\.c\.tblout#\.tblout#g' \
    -e 's#/derived/rust\.#/derived/#g' \
    -e 's#/derived/c\.#/derived/#g' \
    "$input" > "$output"
}

write_tblout_parity() {
  local parity_file="$out_dir/tblout_parity.tsv"
  local case_name rust_tbl c_tbl rust_norm c_norm rust_rows c_rows status

  printf "case\tstatus\trust_rows\tc_rows\n" > "$parity_file"
  for rust_tbl in "$out_dir"/*.rust.tblout; do
    [[ -f "$rust_tbl" ]] || continue
    case_name="$(basename "$rust_tbl" .rust.tblout)"
    c_tbl="$out_dir/${case_name}.c.tblout"
    [[ -f "$c_tbl" ]] || continue

    rust_norm="$out_dir/${case_name}.rust.normalized.tblout"
    c_norm="$out_dir/${case_name}.c.normalized.tblout"
    normalize_tblout_for_parity "$rust_tbl" "$rust_norm"
    normalize_tblout_for_parity "$c_tbl" "$c_norm"
    rust_rows="$(awk 'NF && $1 !~ /^#/ { n++ } END { print n + 0 }' "$rust_tbl")"
    c_rows="$(awk 'NF && $1 !~ /^#/ { n++ } END { print n + 0 }' "$c_tbl")"

    if cmp -s "$rust_norm" "$c_norm"; then
      status="ok"
    else
      status="diff"
      failures=$((failures + 1))
      diff -u "$c_norm" "$rust_norm" > "$out_dir/${case_name}.normalized.diff" || true
    fi
    printf "%s\t%s\t%s\t%s\n" "$case_name" "$status" "$rust_rows" "$c_rows" >> "$parity_file"
  done
}

write_tblout_parity

if [[ "$failures" -ne 0 ]]; then
  echo "$failures realistic Rust/C comparison command(s) failed; see $out_dir/results.tsv" >&2
  exit 1
fi

summary_body="$out_dir/summary.body.tsv"
awk -F '\t' '
  NR == 1 { next }
  {
    key = $1
    impl = $2
    wall[key, impl] = $4
    rss[key, impl] = $7
  }
  END {
    for (k in wall) {
      split(k, parts, SUBSEP)
      cases[parts[1]] = 1
    }
    for (case_name in cases) {
      rw = wall[case_name, "rust"] + 0
      cw = wall[case_name, "c"] + 0
      rr = rss[case_name, "rust"] + 0
      cr = rss[case_name, "c"] + 0
      printf "%s\t%.3f\t%.3f\t%.3f\t%d\t%d\t%.3f\n", case_name, rw, cw, (cw ? rw / cw : 0), rr, cr, (cr ? rr / cr : 0)
    }
  }
' "$out_dir/results.tsv" | sort > "$summary_body"
{
  echo "case	rust_wall_s	c_wall_s	rust_vs_c_wall	rust_rss_kb	c_rss_kb	rust_vs_c_rss"
  cat "$summary_body"
} > "$out_dir/summary.tsv"
rm -f "$summary_body"

echo "realistic Rust/C comparison artifacts written to $out_dir"
