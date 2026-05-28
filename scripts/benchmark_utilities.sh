#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

rust_bin="${RUST_HMMER:-target/release/hmmer}"
c_dir="${C_HMMER_DIR:-hmmer/src}"
rounds="${ROUNDS:-1}"
allow_mismatch="${ALLOW_MISMATCH:-0}"
skip_build="${SKIP_BUILD:-0}"
case_filter="${CASES:-}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${OUT_DIR:-reports/benchmarks/utilities-$stamp}"
all_cases=(
  hmmbuild_20aa
  hmmalign_20aa
  hmmstat_gecco_cluster1
  hmmconvert_gecco_cluster1
  hmmlogo_pkinase
  hmmemit_pkinase_N100
  hmmsim_pkinase_N10000
  alimask_20aa_modelrange
  makehmmerdb_dna_target
  hmmpress_gecco_cluster1
  hmmfetch_gecco_cluster1_valid
)
selected_cases=()

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
    echo "build bundled C HMMER or set C_HMMER_DIR to a directory containing the C HMMER executables" >&2
    return 1
  fi
}

file_size_bytes() {
  wc -c < "$1" | tr -d ' '
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

contains_case() {
  local needle="$1"
  local case_name
  for case_name in "${all_cases[@]}"; do
    if [[ "$case_name" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

validate_cases() {
  local selected
  if [[ -z "$case_filter" ]]; then
    selected_cases=("${all_cases[@]}")
    return 0
  fi

  for selected in ${case_filter//,/ }; do
    if ! contains_case "$selected"; then
      echo "unknown utility benchmark CASES selector: $selected" >&2
      echo "known cases: ${all_cases[*]}" >&2
      return 1
    fi
    selected_cases+=("$selected")
  done

  if [[ "${#selected_cases[@]}" -eq 0 ]]; then
    echo "CASES selected zero utility benchmark cases" >&2
    echo "known cases: ${all_cases[*]}" >&2
    return 1
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

normalize_text() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return
  fi
  REPO_ROOT="$repo_root" OUT_DIR="$out_dir" RUST_BIN="$rust_bin" C_DIR="$c_dir" perl -pe '
    BEGIN {
      @paths = (
        [$ENV{"OUT_DIR"}, "<OUT>"],
        [$ENV{"REPO_ROOT"}, "<REPO>"],
        [$ENV{"RUST_BIN"}, "<RUST_HMMER>"],
        [$ENV{"C_DIR"}, "<C_HMMER_DIR>"],
      );
      @paths = sort { length($b->[0] // "") <=> length($a->[0] // "") } @paths;
    }
    s/\r$//;
    if (/^DATE\s+/ || /^# Date:/ || /^# Current dir:/ || /^# CPU time:/ || /^# Mc\/sec:/) {
      $_ = "";
      next;
    }
    for my $path (@paths) {
      next if !defined $path->[0] || $path->[0] eq "";
      my $quoted = quotemeta($path->[0]);
      s/$quoted/$path->[1]/g;
    }
    s{<OUT>/\S+}{<OUTFILE>}g;
    s/^# Option settings:.*/# Option settings: <NORMALIZED>/;
  ' "$path"
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

run_timed() {
  local prefix="$1"
  shift
  /usr/bin/time -v "$@" > "$out_dir/${prefix}.stdout" 2> "$out_dir/${prefix}.time"
}

append_result() {
  local case_name="$1"
  local impl="$2"
  local round="$3"
  local command_file="$4"
  local run_order="$5"
  local time_file="$out_dir/${case_name}.${impl}.round${round}.time"
  local elapsed user system rss wall_seconds

  elapsed="$(parse_time_field "$time_file" "Elapsed (wall clock) time (h:mm:ss or m:ss)")"
  user="$(parse_time_field "$time_file" "User time (seconds)")"
  system="$(parse_time_field "$time_file" "System time (seconds)")"
  rss="$(parse_time_field "$time_file" "Maximum resident set size (kbytes)")"
  wall_seconds="$(elapsed_to_seconds "$elapsed")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$case_name" "$impl" "$round" "$run_order" "$wall_seconds" "$user" "$system" "$rss" >> "$out_dir/results.tsv"
  {
    echo "case=$case_name"
    echo "impl=$impl"
    echo "round=$round"
    echo "run_order=$run_order"
    echo "command=$(cat "$command_file")"
  } > "$out_dir/${case_name}.${impl}.round${round}.summary"
}

compare_pair() {
  local case_name="$1"
  local round="$2"
  local kind="$3"
  local rust_path="$4"
  local c_path="$5"
  local rust_norm="$out_dir/${case_name}.rust.round${round}.${kind}.normalized"
  local c_norm="$out_dir/${case_name}.c.round${round}.${kind}.normalized"
  local diff_path="$out_dir/${case_name}.round${round}.${kind}.normalized.diff"

  if [[ ! -f "$rust_path" || ! -f "$c_path" ]]; then
    echo "$case_name: missing $kind output for comparison" >&2
    return 1
  fi
  normalize_text "$rust_path" > "$rust_norm"
  normalize_text "$c_path" > "$c_norm"
  sha256sum "$rust_norm" "$c_norm" >> "$out_dir/${case_name}.round${round}.normalized.sha256"
  diff -u "$rust_norm" "$c_norm" > "$diff_path"
}

should_run_case() {
  local case_name="$1"
  local selected
  for selected in "${selected_cases[@]}"; do
    if [[ "$selected" == "$case_name" ]]; then
      return 0
    fi
  done
  return 1
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

record_cargo_config_metadata() {
  local config=".cargo/config.toml"
  local artifact="$out_dir/cargo-config.toml"

  if [[ -f "$config" ]]; then
    cp "$config" "$artifact"
    echo "cargo_config_status=present"
    echo "cargo_config_path=$config"
    echo "cargo_config_bytes=$(file_size_bytes "$config")"
    echo "cargo_config_sha256=$(file_sha256 "$config")"
    echo "cargo_config_artifact=$artifact"
    echo "cargo_config_artifact_sha256=$(file_sha256 "$artifact")"
  else
    echo "cargo_config_status=absent"
    echo "cargo_config_path=$config"
  fi
}

verify_hmmpress_sidecars() {
  local case_name="$1"
  local round="$2"
  local rust_base="$3"
  local c_base="$4"
  local ext impl base path
  local hashes="$out_dir/${case_name}.round${round}.hmmpress_sidecars.sha256"

  : > "$hashes"
  for impl in rust c; do
    if [[ "$impl" == "rust" ]]; then
      base="$rust_base"
    else
      base="$c_base"
    fi
    for ext in h3m h3f h3p h3i; do
      path="${base}.${ext}"
      if [[ ! -s "$path" ]]; then
        echo "$case_name: missing or empty $impl hmmpress sidecar: $path" >&2
        return 1
      fi
      sha256sum "$path" >> "$hashes"
    done
  done

  require_executable "$c_dir/hmmfetch"
  "$rust_bin" fetch "$c_base" "Alpha-amylase" > "$out_dir/${case_name}.rust_fetch_c_pressed.round${round}.stdout"
  "$c_dir/hmmfetch" "$rust_base" "Alpha-amylase" > "$out_dir/${case_name}.c_fetch_rust_pressed.round${round}.stdout"
  "$rust_bin" fetch "$rust_base" "Alpha-amylase" > "$out_dir/${case_name}.rust_fetch_rust_pressed.round${round}.stdout"
  "$c_dir/hmmfetch" "$c_base" "Alpha-amylase" > "$out_dir/${case_name}.c_fetch_c_pressed.round${round}.stdout"
  local failed=0
  compare_pair "$case_name" "$round" "rust_pressed_fetch" \
    "$out_dir/${case_name}.rust_fetch_rust_pressed.round${round}.stdout" \
    "$out_dir/${case_name}.c_fetch_rust_pressed.round${round}.stdout" || failed=1
  compare_pair "$case_name" "$round" "c_pressed_fetch" \
    "$out_dir/${case_name}.rust_fetch_c_pressed.round${round}.stdout" \
    "$out_dir/${case_name}.c_fetch_c_pressed.round${round}.stdout" || failed=1
  return "$failed"
}

write_metadata() {
  local status_file="$out_dir/git-status-short.txt"
  local diff_stat_file="$out_dir/git-diff-stat.txt"
  local diff_file="$out_dir/git-diff.patch"
  local c_tree_root c_status_file c_diff_stat_file

  git status --short 2>/dev/null > "$status_file" || true
  git diff --stat HEAD 2>/dev/null > "$diff_stat_file" || true
  git diff HEAD 2>/dev/null > "$diff_file" || true
  c_tree_root="$(git -C "$c_dir" rev-parse --show-toplevel 2>/dev/null || true)"
  c_status_file="$out_dir/c-hmmer-git-status-short.txt"
  c_diff_stat_file="$out_dir/c-hmmer-git-diff-stat.txt"
  if [[ -n "$c_tree_root" ]]; then
    git -C "$c_tree_root" status --short 2>/dev/null > "$c_status_file" || true
    git -C "$c_tree_root" diff --stat HEAD 2>/dev/null > "$c_diff_stat_file" || true
  else
    : > "$c_status_file"
    : > "$c_diff_stat_file"
  fi

  {
    echo "timestamp_utc=$stamp"
    echo "repo_root=$repo_root"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_short_file=$status_file"
    echo "git_status_short_sha256=$(sha256sum "$status_file" | awk '{print $1}')"
    echo "git_diff_stat_file=$diff_stat_file"
    echo "git_diff_stat_sha256=$(sha256sum "$diff_stat_file" | awk '{print $1}')"
    echo "git_diff_file=$diff_file"
    echo "git_diff_sha256=$(sha256sum "$diff_file" | awk '{print $1}')"
    record_cargo_config_metadata
    echo "cargo=$(cargo --version 2>/dev/null || true)"
    echo "rustc_verbose<<EOF"
    rustc -vV 2>/dev/null || true
    echo "EOF"
    echo "rustflags=${RUSTFLAGS:-}"
    echo "cargo_target_dir=${CARGO_TARGET_DIR:-target}"
    echo "rust_bin=$rust_bin"
    append_path_metadata "rust_bin" "$rust_bin"
    echo "c_hmmer_dir=$c_dir"
    echo "c_hmmer_tree_root=$c_tree_root"
    if [[ -n "$c_tree_root" ]]; then
      echo "c_hmmer_git_commit=$(git -C "$c_tree_root" rev-parse HEAD 2>/dev/null || true)"
      echo "c_hmmer_git_status_short_file=$c_status_file"
      echo "c_hmmer_git_status_short_sha256=$(sha256sum "$c_status_file" | awk '{print $1}')"
      echo "c_hmmer_git_diff_stat_file=$c_diff_stat_file"
      echo "c_hmmer_git_diff_stat_sha256=$(sha256sum "$c_diff_stat_file" | awk '{print $1}')"
    fi
    append_path_metadata "c_hmmbuild" "$c_dir/hmmbuild"
    append_path_metadata "c_hmmalign" "$c_dir/hmmalign"
    append_path_metadata "c_hmmstat" "$c_dir/hmmstat"
    append_path_metadata "c_hmmconvert" "$c_dir/hmmconvert"
    append_path_metadata "c_hmmlogo" "$c_dir/hmmlogo"
    append_path_metadata "c_hmmemit" "$c_dir/hmmemit"
    append_path_metadata "c_hmmsim" "$c_dir/hmmsim"
    append_path_metadata "c_alimask" "$c_dir/alimask"
    append_path_metadata "c_makehmmerdb" "$c_dir/makehmmerdb"
    append_path_metadata "c_hmmpress" "$c_dir/hmmpress"
    append_path_metadata "c_hmmfetch" "$c_dir/hmmfetch"
    echo "rounds=$rounds"
    echo "allow_mismatch=$allow_mismatch"
    echo "case_filter=$case_filter"
    echo "selected_cases=${selected_cases[*]}"
    echo "compare_mode=strict-normalized-diff"
    echo "run_order=odd-rounds-rust-then-c_even-rounds-c-then-rust"
    echo "skip_build=$skip_build"
    echo "uname=$(uname -a)"
    echo "rustc=$(rustc --version 2>/dev/null || true)"
    if [[ -x "$rust_bin" ]]; then
      echo "rust_hmmer_version=$("$rust_bin" --version 2>&1 | head -n 1 || true)"
    fi
    if [[ -x "$c_dir/hmmsearch" ]]; then
      echo "c_hmmer_hmmsearch_banner=$("$c_dir/hmmsearch" -h 2>&1 | head -n 1 || true)"
      echo "c_hmmer_hmmsearch_version=$("$c_dir/hmmsearch" -h 2>&1 | sed -n '2p' || true)"
    fi
  } > "$out_dir/metadata.txt"
}

run_case() {
  local case_name="$1"
  local c_tool="$2"
  local rust_subcmd="$3"
  shift 3
  local compare_kinds=("$@")

  require_executable "$rust_bin"
  require_executable "$c_dir/$c_tool"

  for round in $(seq 1 "$rounds"); do
    local rust_cmd="$out_dir/${case_name}.rust.round${round}.cmd"
    local c_cmd="$out_dir/${case_name}.c.round${round}.cmd"
    local rust_stdout="$out_dir/${case_name}.rust.round${round}.stdout"
    local c_stdout="$out_dir/${case_name}.c.round${round}.stdout"
    local rust_out="$out_dir/${case_name}.rust.round${round}.out"
    local c_out="$out_dir/${case_name}.c.round${round}.out"
    local rust_args=("$rust_subcmd")
    local c_args=()

    case "$case_name" in
      hmmbuild_20aa)
        require_file "hmmer/testsuite/20aa.sto"
        record_dataset "$case_name input" "hmmer/testsuite/20aa.sto"
        rust_args+=("$rust_out" "hmmer/testsuite/20aa.sto")
        c_args+=("$c_out" "hmmer/testsuite/20aa.sto")
        ;;
      hmmalign_20aa)
        require_file "hmmer/testsuite/20aa.hmm"
        require_file "hmmer/testsuite/20aa-alitest.fa"
        record_dataset "$case_name hmm" "hmmer/testsuite/20aa.hmm"
        record_dataset "$case_name seqs" "hmmer/testsuite/20aa-alitest.fa"
        rust_args+=("hmmer/testsuite/20aa.hmm" "hmmer/testsuite/20aa-alitest.fa")
        c_args+=("hmmer/testsuite/20aa.hmm" "hmmer/testsuite/20aa-alitest.fa")
        ;;
      hmmstat_gecco_cluster1|hmmconvert_gecco_cluster1)
        require_file "test_data/gecco_cluster1_hmms.hmm"
        record_dataset "$case_name hmm" "test_data/gecco_cluster1_hmms.hmm"
        rust_args+=("test_data/gecco_cluster1_hmms.hmm")
        c_args+=("test_data/gecco_cluster1_hmms.hmm")
        ;;
      hmmlogo_pkinase)
        require_file "test_data/Pkinase_pfam.hmm"
        record_dataset "$case_name hmm" "test_data/Pkinase_pfam.hmm"
        rust_args+=("test_data/Pkinase_pfam.hmm")
        c_args+=("test_data/Pkinase_pfam.hmm")
        ;;
      hmmemit_pkinase_N100)
        require_file "test_data/Pkinase_pfam.hmm"
        record_dataset "$case_name hmm" "test_data/Pkinase_pfam.hmm"
        rust_args+=("--seed" "42" "-N" "100" "test_data/Pkinase_pfam.hmm")
        c_args+=("--seed" "42" "-N" "100" "test_data/Pkinase_pfam.hmm")
        ;;
      hmmsim_pkinase_N10000)
        require_file "test_data/Pkinase_pfam.hmm"
        record_dataset "$case_name hmm" "test_data/Pkinase_pfam.hmm"
        rust_args+=("--seed" "42" "-N" "10000" "--msv" "test_data/Pkinase_pfam.hmm")
        c_args+=("--seed" "42" "-N" "10000" "--msv" "test_data/Pkinase_pfam.hmm")
        ;;
      alimask_20aa_modelrange)
        require_file "hmmer/testsuite/20aa.sto"
        record_dataset "$case_name msa" "hmmer/testsuite/20aa.sto"
        rust_args+=("--modelrange" "2..10" "hmmer/testsuite/20aa.sto" "$rust_out")
        c_args+=("--modelrange" "2..10" "hmmer/testsuite/20aa.sto" "$c_out")
        ;;
      makehmmerdb_dna_target)
        require_file "hmmer/tutorial/dna_target.fa"
        record_dataset "$case_name seqs" "hmmer/tutorial/dna_target.fa"
        rust_args+=("hmmer/tutorial/dna_target.fa" "$rust_out")
        c_args+=("hmmer/tutorial/dna_target.fa" "$c_out")
        ;;
      hmmpress_gecco_cluster1)
        require_file "test_data/gecco_cluster1_hmms.hmm"
        record_dataset "$case_name hmm" "test_data/gecco_cluster1_hmms.hmm"
        cp "test_data/gecco_cluster1_hmms.hmm" "$rust_out.hmm"
        cp "test_data/gecco_cluster1_hmms.hmm" "$c_out.hmm"
        rust_args+=("-f" "$rust_out.hmm")
        c_args+=("-f" "$c_out.hmm")
        ;;
      hmmfetch_gecco_cluster1_valid)
        require_file "test_data/gecco_cluster1_hmms.hmm"
        record_dataset "$case_name hmm" "test_data/gecco_cluster1_hmms.hmm"
        rust_args+=("test_data/gecco_cluster1_hmms.hmm" "Alpha-amylase")
        c_args+=("test_data/gecco_cluster1_hmms.hmm" "Alpha-amylase")
        ;;
      *)
        echo "unknown utility benchmark case: $case_name" >&2
        return 1
        ;;
    esac

    printf "%q " "$rust_bin" "${rust_args[@]}" > "$rust_cmd"
    printf "%q " "$c_dir/$c_tool" "${c_args[@]}" > "$c_cmd"
    local first_impl="rust"
    local second_impl="c"
    local run_order="rust,c"
    if (( round % 2 == 0 )); then
      first_impl="c"
      second_impl="rust"
      run_order="c,rust"
    fi
    printf "%s\t%s\t%s\t%s\n" "$case_name" "$round" "$first_impl" "$second_impl" >> "$out_dir/run_order.tsv"

    for impl in "$first_impl" "$second_impl"; do
      case "$impl" in
        rust)
          echo "running $case_name round $round: Rust"
          run_timed "${case_name}.rust.round${round}" "$rust_bin" "${rust_args[@]}"
          ;;
        c)
          echo "running $case_name round $round: C"
          run_timed "${case_name}.c.round${round}" "$c_dir/$c_tool" "${c_args[@]}"
          ;;
      esac
    done
    append_result "$case_name" "rust" "$round" "$rust_cmd" "$run_order"
    append_result "$case_name" "c" "$round" "$c_cmd" "$run_order"

    : > "$out_dir/${case_name}.round${round}.normalized.sha256"
    local kind failed=0
    if [[ "$case_name" == "hmmpress_gecco_cluster1" ]]; then
      verify_hmmpress_sidecars "$case_name" "$round" "$rust_out.hmm" "$c_out.hmm" || failed=1
    fi
    for kind in "${compare_kinds[@]}"; do
      case "$kind" in
        stdout)
          compare_pair "$case_name" "$round" "stdout" "$rust_stdout" "$c_stdout" || failed=1
          ;;
        primary)
          compare_pair "$case_name" "$round" "primary" "$rust_out" "$c_out" || failed=1
          ;;
      esac
    done
    if [[ "$failed" == "1" && "$allow_mismatch" != "1" ]]; then
      return 1
    fi
  done
}

validate_cases
mkdir -p "$out_dir"
if [[ "$skip_build" != "1" ]]; then
  cargo build --release
fi
write_metadata
printf "case\timpl\tround\trun_order\twall_seconds\tuser_seconds\tsystem_seconds\tmax_rss_kb\n" > "$out_dir/results.tsv"
printf "case\tround\tfirst_impl\tsecond_impl\n" > "$out_dir/run_order.tsv"
printf "case_dataset\tpath\tstatus\tbytes\tsha256\n" > "$out_dir/datasets.tsv"

if should_run_case "hmmbuild_20aa"; then run_case "hmmbuild_20aa" "hmmbuild" "build" "stdout" "primary"; fi
if should_run_case "hmmalign_20aa"; then run_case "hmmalign_20aa" "hmmalign" "align" "stdout"; fi
if should_run_case "hmmstat_gecco_cluster1"; then run_case "hmmstat_gecco_cluster1" "hmmstat" "stat" "stdout"; fi
if should_run_case "hmmconvert_gecco_cluster1"; then run_case "hmmconvert_gecco_cluster1" "hmmconvert" "convert" "stdout"; fi
if should_run_case "hmmlogo_pkinase"; then run_case "hmmlogo_pkinase" "hmmlogo" "logo" "stdout"; fi
if should_run_case "hmmemit_pkinase_N100"; then run_case "hmmemit_pkinase_N100" "hmmemit" "emit" "stdout"; fi
if should_run_case "hmmsim_pkinase_N10000"; then run_case "hmmsim_pkinase_N10000" "hmmsim" "sim"; fi
if should_run_case "alimask_20aa_modelrange"; then run_case "alimask_20aa_modelrange" "alimask" "alimask" "stdout" "primary"; fi
if should_run_case "makehmmerdb_dna_target"; then run_case "makehmmerdb_dna_target" "makehmmerdb" "makehmmerdb"; fi
if should_run_case "hmmpress_gecco_cluster1"; then run_case "hmmpress_gecco_cluster1" "hmmpress" "press" "stdout"; fi
if should_run_case "hmmfetch_gecco_cluster1_valid"; then run_case "hmmfetch_gecco_cluster1_valid" "hmmfetch" "fetch" "stdout"; fi

echo "utility benchmark artifacts written to $out_dir"
