#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rust_bin="${RUST_HMMER:-target/release/hmmer}"
c_dir="${C_HMMER_DIR:-hmmer/src}"
threads="${THREADS:-1}"
rounds="${ROUNDS:-1}"
top_n="${TOP_N:-20}"
allow_mismatch="${ALLOW_MISMATCH:-0}"
skip_build="${SKIP_BUILD:-0}"
case_filter="${CASES:-}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${OUT_DIR:-reports/benchmarks/$stamp}"
mkdir -p "$out_dir"

if [[ "$skip_build" != "1" ]]; then
  cargo build --release
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    return 1
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "missing required executable: $path" >&2
    return 1
  fi
}

decompress_cat() {
  local path="$1"
  case "$path" in
    *.gz) gzip -dc "$path" ;;
    *) cat "$path" ;;
  esac
}

fasta_stats() {
  local path="$1"
  decompress_cat "$path" | awk '
    /^>/ { seqs++; next }
    { gsub(/[[:space:]]/, ""); residues += length($0) }
    END { printf "%d\t%d", seqs, residues }
  '
}

ensure_query_by_id() {
  local dataset="$1"
  local query_id="$2"
  local output="$3"
  if [[ -s "$output" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  decompress_cat "$dataset" | awk -v id="$query_id" '
    /^>/ {
      hit = (index($0, id) > 0)
      if (hit) print
      next
    }
    hit { print }
  ' > "$output"
  if [[ ! -s "$output" ]]; then
    echo "could not extract query id '$query_id' from $dataset" >&2
    return 1
  fi
}

count_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  awk 'NF && $1 !~ /^#/' "$path" | wc -l | tr -d ' '
}

top_targets() {
  local path="$1"
  local n="$2"
  if [[ ! -f "$path" ]]; then
    return
  fi
  awk -v limit="$n" '
    NF && $1 !~ /^#/ {
      print $1
      count++
      if (count >= limit) {
        exit
      }
    }
  ' "$path"
}

tblout_core_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return
  fi
  awk '
    NF && $1 !~ /^#/ {
      limit = (NF < 18 ? NF : 18)
      row = $1
      for (i = 2; i <= limit; i++) {
        row = row "\t" $i
      }
      print row
    }
  ' "$path"
}

domtblout_core_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return
  fi
  awk '
    NF && $1 !~ /^#/ {
      limit = (NF < 22 ? NF : 22)
      row = $1
      for (i = 2; i <= limit; i++) {
        row = row "\t" $i
      }
      print row
    }
  ' "$path"
}

normalize_benchmark_output() {
  local path="$1"
  local tool="$2"
  if [[ ! -f "$path" ]]; then
    return
  fi

  REPO_ROOT="$repo_root" \
  OUT_DIR="$out_dir" \
  RUST_BIN="$rust_bin" \
  C_BIN="$c_dir/$tool" \
  C_DIR="$c_dir" \
  perl -pe '
    BEGIN {
      @paths = (
        [$ENV{"OUT_DIR"}, "<OUT>"],
        [$ENV{"REPO_ROOT"}, "<REPO>"],
        [$ENV{"RUST_BIN"}, "<RUST_HMMER>"],
        [$ENV{"C_BIN"}, "<C_HMMER>"],
        [$ENV{"C_DIR"}, "<C_HMMER_DIR>"],
      );
      @paths = sort { length($b->[0] // "") <=> length($a->[0] // "") } @paths;
    }
    s/\r$//;
    s/^# Date:\s+.*/# Date:            <DATE>/;
    s/^# Current dir:\s+.*/# Current dir:     <CWD>/;
    for my $path (@paths) {
      next if !defined $path->[0] || $path->[0] eq "";
      my $quoted = quotemeta($path->[0]);
      s/$quoted/$path->[1]/g;
    }
    s/^# Option settings:\s+(?:<RUST_HMMER>(?:\s+\S+)?|<C_HMMER>)\s+/# Option settings: <HMMER_COMMAND> /;
    s{<OUT>/\S+\.stdout}{<STDOUT>}g;
    s{<OUT>/\S+\.tblout}{<TBLOUT>}g;
    s{<OUT>/\S+\.domtblout}{<DOMTBLOUT>}g;
    s{<OUT>/\S+}{<OUTFILE>}g;
  ' "$path"
}

write_normalized_artifact() {
  local case_name="$1"
  local round="$2"
  local impl="$3"
  local kind="$4"
  local path="$5"
  local tool="$6"
  local normalized="$out_dir/${case_name}.${impl}.round${round}.${kind}.normalized"

  if [[ ! -f "$path" ]]; then
    return
  fi

  normalize_benchmark_output "$path" "$tool" > "$normalized"
  sha256sum "$normalized" > "$normalized.sha256"
}

append_checksum_manifest() {
  local case_name="$1"
  local round="$2"
  local manifest="$out_dir/${case_name}.round${round}.normalized.sha256"

  : > "$manifest"
  for artifact in \
    "$out_dir/${case_name}.rust.round${round}.stdout.normalized" \
    "$out_dir/${case_name}.c.round${round}.stdout.normalized" \
    "$out_dir/${case_name}.rust.round${round}.tblout.normalized" \
    "$out_dir/${case_name}.c.round${round}.tblout.normalized" \
    "$out_dir/${case_name}.rust.round${round}.domtblout.normalized" \
    "$out_dir/${case_name}.c.round${round}.domtblout.normalized"; do
    if [[ -f "$artifact" ]]; then
      sha256sum "$artifact" >> "$manifest"
    fi
  done
}

compare_normalized_pair() {
  local case_name="$1"
  local round="$2"
  local kind="$3"
  local rust_path="$4"
  local c_path="$5"
  local tool="$6"
  local diff_path="$out_dir/${case_name}.round${round}.${kind}.full.normalized.diff"

  if [[ ! -f "$rust_path" && ! -f "$c_path" ]]; then
    return 0
  fi
  if [[ ! -f "$rust_path" || ! -f "$c_path" ]]; then
    echo "$case_name: missing $kind output for full normalized comparison: Rust=$rust_path C=$c_path" >&2
    return 1
  fi

  write_normalized_artifact "$case_name" "$round" "rust" "$kind" "$rust_path" "$tool"
  write_normalized_artifact "$case_name" "$round" "c" "$kind" "$c_path" "$tool"

  if ! diff -u \
    "$out_dir/${case_name}.rust.round${round}.${kind}.normalized" \
    "$out_dir/${case_name}.c.round${round}.${kind}.normalized" \
    > "$diff_path"; then
    echo "$case_name: normalized full $kind output differs; see ${case_name}.round${round}.${kind}.full.normalized.diff" >&2
    return 1
  fi
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
      if (n == 1) {
        print a[1] + 0
      } else if (n == 2) {
        print (a[1] * 60) + a[2]
      } else {
        print (a[1] * 3600) + (a[2] * 60) + a[3]
      }
    }
  '
}

write_metadata() {
  {
    echo "timestamp_utc=$stamp"
    echo "repo_root=$repo_root"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_sha256=$(git status --short 2>/dev/null | sha256sum | awk '{print $1}')"
    echo "rust_bin=$rust_bin"
    echo "c_hmmer_dir=$c_dir"
    echo "threads=$threads"
    echo "rounds=$rounds"
    echo "top_n=$top_n"
    echo "uname=$(uname -a)"
    echo "rustc=$(rustc --version 2>/dev/null || true)"
    if [[ -x "$rust_bin" ]]; then
      echo "rust_hmmer_version=$("$rust_bin" --version 2>&1 | head -n 1 || true)"
    fi
    if [[ -x "$c_dir/hmmsearch" ]]; then
      echo "c_hmmsearch_version=$("$c_dir/hmmsearch" -h 2>&1 | head -n 2 | tr '\n' ' ' || true)"
    fi
  } > "$out_dir/metadata.txt"
}

run_timed() {
  local prefix="$1"
  shift
  local stdout_path="$out_dir/${prefix}.stdout"
  local stderr_path="$out_dir/${prefix}.time"
  /usr/bin/time -v "$@" > "$stdout_path" 2> "$stderr_path"
}

append_result() {
  local case_name="$1"
  local impl="$2"
  local round="$3"
  local table="$4"
  local domtable="$5"
  local time_file="$6"
  local command_file="$7"

  local elapsed user system rss wall_seconds rows dom_rows
  elapsed="$(parse_time_field "$time_file" "Elapsed (wall clock) time (h:mm:ss or m:ss)")"
  user="$(parse_time_field "$time_file" "User time (seconds)")"
  system="$(parse_time_field "$time_file" "System time (seconds)")"
  rss="$(parse_time_field "$time_file" "Maximum resident set size (kbytes)")"
  wall_seconds="$(elapsed_to_seconds "$elapsed")"
  rows="$(count_rows "$table")"
  dom_rows="$(count_rows "$domtable")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$case_name" "$impl" "$round" "$threads" "$wall_seconds" "$user" "$system" "$rss" "$rows" "$dom_rows" \
    >> "$out_dir/results.tsv"
  {
    echo "case=$case_name"
    echo "impl=$impl"
    echo "round=$round"
    echo "command=$(cat "$command_file")"
    echo "tblout_rows=$rows"
    echo "domtblout_rows=$dom_rows"
    echo "top_tblout_targets:"
    top_targets "$table" "$top_n"
  } > "$out_dir/${case_name}.${impl}.round${round}.summary"
}

compare_outputs() {
  local case_name="$1"
  local round="$2"
  local tool="$3"
  local rust_stdout="$4"
  local c_stdout="$5"
  local rust_tbl="$6"
  local c_tbl="$7"
  local rust_dom="$8"
  local c_dom="$9"
  local rust_rows c_rows rust_dom_rows c_dom_rows
  rust_rows="$(count_rows "$rust_tbl")"
  c_rows="$(count_rows "$c_tbl")"
  rust_dom_rows="$(count_rows "$rust_dom")"
  c_dom_rows="$(count_rows "$c_dom")"

  local failed=0
  if [[ "$rust_rows" != "$c_rows" ]]; then
    echo "$case_name: tblout row mismatch: Rust=$rust_rows C=$c_rows" >&2
    failed=1
  fi
  if [[ "$rust_dom_rows" != "$c_dom_rows" ]]; then
    echo "$case_name: domtblout row mismatch: Rust=$rust_dom_rows C=$c_dom_rows" >&2
    failed=1
  fi
  if ! diff -u <(top_targets "$rust_tbl" "$top_n") <(top_targets "$c_tbl" "$top_n") > "$out_dir/${case_name}.top_targets.diff"; then
    echo "$case_name: top $top_n tblout target order differs; see ${case_name}.top_targets.diff" >&2
    failed=1
  fi
  if ! diff -u <(tblout_core_rows "$rust_tbl") <(tblout_core_rows "$c_tbl") > "$out_dir/${case_name}.tblout.core.diff"; then
    echo "$case_name: tblout core rows differ; see ${case_name}.tblout.core.diff" >&2
    failed=1
  fi
  if [[ -f "$rust_dom" || -f "$c_dom" ]]; then
    if ! diff -u <(domtblout_core_rows "$rust_dom") <(domtblout_core_rows "$c_dom") > "$out_dir/${case_name}.domtblout.core.diff"; then
      echo "$case_name: domtblout core rows differ; see ${case_name}.domtblout.core.diff" >&2
      failed=1
    fi
  fi
  if ! compare_normalized_pair "$case_name" "$round" "stdout" "$rust_stdout" "$c_stdout" "$tool"; then
    failed=1
  fi
  if ! compare_normalized_pair "$case_name" "$round" "tblout" "$rust_tbl" "$c_tbl" "$tool"; then
    failed=1
  fi
  if ! compare_normalized_pair "$case_name" "$round" "domtblout" "$rust_dom" "$c_dom" "$tool"; then
    failed=1
  fi
  append_checksum_manifest "$case_name" "$round"
  if [[ "$failed" == "1" && "$allow_mismatch" != "1" ]]; then
    return 1
  fi
}

run_case() {
  local case_name="$1"
  local tool="$2"
  local query="$3"
  local target="$4"
  local extra="$5"

  require_file "$query"
  require_file "$target"
  require_executable "$rust_bin"
  require_executable "$c_dir/$tool"

  local stats
  stats="$(fasta_stats "$target")"
  echo "$case_name target_stats	$target	$stats" >> "$out_dir/datasets.tsv"

  local rust_tool_args=()
  if [[ "$tool" == "hmmsearch" ]]; then
    rust_tool_args=("search")
  else
    rust_tool_args=("$tool")
  fi

  for round in $(seq 1 "$rounds"); do
    local rust_tbl="$out_dir/${case_name}.rust.round${round}.tblout"
    local c_tbl="$out_dir/${case_name}.c.round${round}.tblout"
    local rust_dom="$out_dir/${case_name}.rust.round${round}.domtblout"
    local c_dom="$out_dir/${case_name}.c.round${round}.domtblout"
    local rust_cmd="$out_dir/${case_name}.rust.round${round}.cmd"
    local c_cmd="$out_dir/${case_name}.c.round${round}.cmd"
    local rust_stdout="$out_dir/${case_name}.rust.round${round}.stdout"
    local c_stdout="$out_dir/${case_name}.c.round${round}.stdout"

    local rust_args=("${rust_tool_args[@]}")
    local c_args=()
    read -r -a extra_args <<< "$extra"

    case "$tool" in
      hmmsearch)
        rust_args+=("--cpu" "$threads" "--noali" "--tblout" "$rust_tbl" "--domtblout" "$rust_dom")
        c_args+=("--cpu" "$threads" "--noali" "--tblout" "$c_tbl" "--domtblout" "$c_dom")
        ;;
      phmmer)
        rust_args+=("--cpu" "$threads" "--tblout" "$rust_tbl")
        c_args+=("--cpu" "$threads" "--tblout" "$c_tbl")
        ;;
      jackhmmer)
        rust_args+=("--cpu" "$threads" "--tblout" "$rust_tbl" "--domtblout" "$rust_dom")
        c_args+=("--cpu" "$threads" "--tblout" "$c_tbl" "--domtblout" "$c_dom")
        ;;
      *)
        echo "unsupported benchmark tool: $tool" >&2
        return 1
        ;;
    esac

    rust_args+=("${extra_args[@]}" "$query" "$target")
    c_args+=("${extra_args[@]}" "$query" "$target")

    printf "%q " "$rust_bin" "${rust_args[@]}" > "$rust_cmd"
    printf "%q " "$c_dir/$tool" "${c_args[@]}" > "$c_cmd"

    echo "running $case_name round $round: Rust"
    run_timed "${case_name}.rust.round${round}" "$rust_bin" "${rust_args[@]}"
    echo "running $case_name round $round: C"
    run_timed "${case_name}.c.round${round}" "$c_dir/$tool" "${c_args[@]}"

    append_result "$case_name" "rust" "$round" "$rust_tbl" "$rust_dom" "$out_dir/${case_name}.rust.round${round}.time" "$rust_cmd"
    append_result "$case_name" "c" "$round" "$c_tbl" "$c_dom" "$out_dir/${case_name}.c.round${round}.time" "$c_cmd"
    compare_outputs "$case_name" "$round" "$tool" "$rust_stdout" "$c_stdout" "$rust_tbl" "$c_tbl" "$rust_dom" "$c_dom"
  done
}

should_run_case() {
  local case_name="$1"
  if [[ -z "$case_filter" ]]; then
    return 0
  fi
  local normalized="${case_filter//,/ }"
  local selected
  for selected in $normalized; do
    if [[ "$selected" == "$case_name" ]]; then
      return 0
    fi
  done
  return 1
}

write_metadata
printf "case\timpl\tround\tthreads\twall_seconds\tuser_seconds\tsystem_seconds\tmax_rss_kb\ttblout_rows\tdomtblout_rows\n" > "$out_dir/results.tsv"
printf "case_dataset\tpath\tsequences\tresidues\n" > "$out_dir/datasets.tsv"

medium_protein="external/protein_medium/uniprot_UP000005640_human.fasta.gz"
large_protein="external/protein_large/uniprot_sprot.fasta.gz"
medium_protein_rewindable="${JACKHMMER_MEDIUM_TARGET:-external/protein_medium/uniprot_UP000005640_human.fasta}"
medium_query="external/protein_medium/queries/sp_O43739_CYH3_HUMAN.fa"
large_query="external/protein_large/queries/sp_O43739_CYH3_HUMAN.fa"

if [[ -f "$medium_protein" ]]; then
  ensure_query_by_id "$medium_protein" "sp|O43739|CYH3_HUMAN" "$medium_query"
fi
if [[ -f "$large_protein" ]]; then
  ensure_query_by_id "$large_protein" "sp|O43739|CYH3_HUMAN" "$large_query"
fi

if should_run_case "hmmsearch_human_pkinase"; then
  run_case "hmmsearch_human_pkinase" "hmmsearch" "test_data/Pkinase_pfam.hmm" "$medium_protein" ""
fi
if should_run_case "hmmsearch_sprot_pkinase"; then
  run_case "hmmsearch_sprot_pkinase" "hmmsearch" "test_data/Pkinase_pfam.hmm" "$large_protein" ""
fi
if should_run_case "phmmer_human_cyh3"; then
  run_case "phmmer_human_cyh3" "phmmer" "$medium_query" "$medium_protein" ""
fi
if should_run_case "jackhmmer_human_cyh3_N2"; then
  run_case "jackhmmer_human_cyh3_N2" "jackhmmer" "$medium_query" "$medium_protein_rewindable" "-N 2"
fi

echo "benchmark artifacts written to $out_dir"
