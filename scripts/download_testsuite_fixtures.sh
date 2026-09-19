#!/usr/bin/env bash
# Regenerate the non-upstream HMM fixtures that the test suite expects under
# hmmer/testsuite/ (which is a gitignored snapshot of upstream HMMER, so these
# files are absent from a clean checkout).
#
# Every HMM fixture here is a plain `hmmfetch -f` of Pfam families from a
# pinned Pfam release, verified byte-for-byte against the original files on
# 2026-09-19:
#
#   gecco_pfam5.hmm, gecco_missed*_hmms.hmm   Pfam 35.0
#   minipfam.hmm                               Pfam 34.0 (its first ten families)
#
# minipfam.hmm was originally written by HMMER 3.3.2, so the regenerated copy
# differs from the original only in the ten "HMMER3/f [version]" lines.
#
# NOT covered: gecco_*_proteins.faa / gecco_proteins.faa. Those are gene
# predictions on GenBank CP157504.1 produced by GECCO's pipeline and have no
# downloadable source, so they cannot be regenerated here.
#
# Usage:
#   scripts/download_testsuite_fixtures.sh
#   FIXTURE_ROOT=/big/cache REFRESH=1 scripts/download_testsuite_fixtures.sh
#
# The ~290 MB Pfam-A.hmm.gz archives (and their ~1.5 GB decompressed forms) are
# cached under $FIXTURE_ROOT (default external/pfam_releases) and reused.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fixture_root="${FIXTURE_ROOT:-external/pfam_releases}"
testsuite_dir="${TESTSUITE_DIR:-hmmer/testsuite}"
refresh="${REFRESH:-0}"

mkdir -p "$fixture_root" "$testsuite_dir"

pfam_release_url() {
  echo "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam$1/Pfam-A.hmm.gz"
}

# sha256 of the release archives as served by EBI.
declare -A pfam_archive_sha256=(
  [35.0]=48ec2d1123c84046b00279eae1fb3d5be1b578e6221453f329d16954c89d0d35
  [34.0]=b18a98bb1f92b9afb7533fda43baac98d9c69e5b97e68592a43fc00d671ea3f6
)

# sha256 of each regenerated fixture.
declare -A fixture_sha256=(
  [gecco_pfam5.hmm]=951f49f3526ae6e524375ee68f5e04c151af74e8ac220d39f2756d852d0abbdb
  [gecco_missed_hmms.hmm]=2a4141670b10287926fbf1332225949357f7c80feb9725bfe4b754607ef5edb8
  [gecco_missed2_hmms.hmm]=a11cc99fbb97f7eaf3a8f697bdf03288b6a1c5c5ccdc0acbe9bf04d4208af1d4
  [gecco_missed3_hmms.hmm]=32a4ce8553d7d47afaf1440a288df2888e4e33d50c2f9430c5ab0883fd5a0559
  [gecco_missed4_hmms.hmm]=87fc8870bc0f183449e0a19e090228d0098b7de1bc4104b5819e7669a99afee9
  [minipfam.hmm]=29db0c48160a53eea54589151e6b247a1d29f4a63fc51ece4dfae0ac15011e5f
)

# Pfam accessions (with version) making up each fixture, in file order.
declare -A fixture_keys=(
  [gecco_pfam5.hmm]="PF00004.32 PF13191.9 PF13304.9 PF13555.9 PF02463.22"
  [gecco_missed_hmms.hmm]="PF13175.9 PF13238.9 PF00561.23 PF00702.29 PF03023.17 PF02378.21 PF01547.28 PF00083.27"
  [gecco_missed2_hmms.hmm]="PF07728.17 PF13508.10 PF13523.9 PF01590.29 PF08242.15 PF12832.10 PF01546.31 PF02719.18 PF02463.22 PF02223.20"
  [gecco_missed3_hmms.hmm]="PF01073.22 PF13191.9 PF13673.10 PF13302.10 PF13420.10 PF13527.10 PF01467.29 PF01266.27 PF06808.15 PF03807.20 PF02525.20 PF00462.27 PF18029.4 PF13443.9 PF13460.9 PF13188.10 PF02129.21 PF13531.9 PF01118.27 PF01751.25"
  [gecco_missed4_hmms.hmm]="PF13304.9 PF13476.9 PF13479.9 PF00583.28 PF13444.9 PF02558.19 PF01869.23 PF01256.20 PF02353.23 PF01370.24 PF08445.13 PF01728.22 PF08279.15 PF13412.9 PF01381.25 PF12802.10 PF08241.15 PF08003.14 PF01926.26 PF13454.9 PF01235.20 PF09084.14 PF03575.20 PF03848.17 PF13098.9 PF06609.16 PF02254.21 PF01978.22 PF01336.28 PF13519.9"
  [minipfam.hmm]="PF10417.11 PF12574.10 PF09847.11 PF00244.22 PF16998.7 PF00389.32 PF02826.21 PF00198.25 PF16078.7 PF04029.16"
)
declare -A fixture_release=(
  [gecco_pfam5.hmm]=35.0
  [gecco_missed_hmms.hmm]=35.0
  [gecco_missed2_hmms.hmm]=35.0
  [gecco_missed3_hmms.hmm]=35.0
  [gecco_missed4_hmms.hmm]=35.0
  [minipfam.hmm]=34.0
)

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

check_sha256() {
  local path="$1" expected="$2"
  local actual
  actual="$(sha256_of "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "checksum mismatch for $path" >&2
    echo "  expected $expected" >&2
    echo "  actual   $actual" >&2
    return 1
  fi
}

download_file() {
  local url="$1" output="$2" expected_sha="$3"
  local tmp="${output}.tmp"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
  else
    echo "downloading $url"
    curl -fL --http1.1 --retry 3 --retry-delay 2 -o "$tmp" "$url"
    mv "$tmp" "$output"
  fi
  check_sha256 "$output" "$expected_sha"
}

decompress_gzip() {
  local input="$1" output="$2"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    echo "using existing $output"
    return 0
  fi
  echo "decompressing $input"
  gzip -dc "$input" > "${output}.tmp"
  mv "${output}.tmp" "$output"
}

# Prefer the bundled C hmmfetch; fall back to this port's `hmmer fetch`, which
# is verified byte-identical on these fixtures.
hmmfetch_cmd() {
  if [[ -x hmmer/src/hmmfetch ]]; then
    echo "hmmer/src/hmmfetch"
  elif [[ -x target/release/hmmer ]]; then
    echo "target/release/hmmer fetch"
  elif [[ -x target/debug/hmmer ]]; then
    echo "target/debug/hmmer fetch"
  else
    echo "no hmmfetch found: build hmmer/src/hmmfetch or run 'cargo build --release'" >&2
    return 1
  fi
}

fetch_fixture() {
  local name="$1"
  local release="${fixture_release[$name]}"
  local output="$testsuite_dir/$name"
  local keyfile="$fixture_root/$name.keys"
  local pfam_hmm="$fixture_root/Pfam-A.$release.hmm"

  if [[ -s "$output" && "$refresh" != "1" ]]; then
    if check_sha256 "$output" "${fixture_sha256[$name]}" 2>/dev/null; then
      echo "using existing $output"
      return 0
    fi
    echo "regenerating $output (checksum differs from the pinned fixture)"
  fi

  tr ' ' '\n' <<< "${fixture_keys[$name]}" > "$keyfile"
  echo "fetching $(wc -l < "$keyfile") families from Pfam $release into $output"
  # shellcheck disable=SC2046
  $(hmmfetch_cmd) -f "$pfam_hmm" "$keyfile" > "${output}.tmp"
  mv "${output}.tmp" "$output"
  check_sha256 "$output" "${fixture_sha256[$name]}"
}

hmmfetch_cmd > /dev/null

for release in 35.0 34.0; do
  download_file "$(pfam_release_url "$release")" "$fixture_root/Pfam-A.$release.hmm.gz" "${pfam_archive_sha256[$release]}"
  decompress_gzip "$fixture_root/Pfam-A.$release.hmm.gz" "$fixture_root/Pfam-A.$release.hmm"
done

for name in gecco_pfam5.hmm gecco_missed_hmms.hmm gecco_missed2_hmms.hmm gecco_missed3_hmms.hmm gecco_missed4_hmms.hmm minipfam.hmm; do
  fetch_fixture "$name"
done

missing=()
for faa in gecco_proteins.faa gecco_missed_proteins.faa gecco_missed2_proteins.faa gecco_missed3_proteins.faa gecco_missed4_proteins.faa; do
  [[ -s "$testsuite_dir/$faa" ]] || missing+=("$faa")
done
if (( ${#missing[@]} )); then
  echo
  echo "NOTE: the following protein fixtures have no downloadable source and are still missing:"
  for faa in "${missing[@]}"; do
    echo "  $testsuite_dir/$faa"
  done
  echo "They are GECCO gene predictions on GenBank CP157504.1; see REAL_WORLD_FIXTURES.md."
fi

echo "testsuite HMM fixtures ready in $testsuite_dir"
