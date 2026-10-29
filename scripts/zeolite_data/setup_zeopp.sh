#!/usr/bin/env bash
set -euo pipefail

prefix=${1:-"$HOME/opt/zeopp-0.3-grid015"}
prefix=$(mkdir -p "$prefix" && cd "$prefix" && pwd)
archive="$prefix/zeo++-0.3.tar.gz"
source_dir="$prefix/zeo++-0.3"
build_jobs=${BUILD_JOBS:-2}

if [[ ! -d "$source_dir" ]]; then
  if [[ ! -s "$archive" ]]; then
    command -v curl >/dev/null || { echo "curl is required to download Zeo++." >&2; exit 1; }
    temporary="$archive.part"
    rm -f "$temporary"
    curl --fail --location --retry 3 --output "$temporary" \
      "https://www.zeoplusplus.org/zeo++-0.3.tar.gz"
    mv "$temporary" "$archive"
  fi
  tar -xzf "$archive" -C "$prefix"
fi

[[ -d "$source_dir/voro++/src" ]] || {
  echo "Expected the official Zeo++ 0.3 source layout at $source_dir." >&2
  exit 1
}
[[ -f "$source_dir/grid.cc" ]] || {
  echo "Could not find grid.cc under $source_dir." >&2
  exit 1
}

python3 - "$source_dir/grid.cc" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text()
updated, replacements = re.subn(
    r"(?m)^(\s*#define\s+GRIDRES\s+)[0-9.eE+-]+",
    r"\g<1>0.15",
    source,
)
if replacements != 1:
    raise SystemExit(f"Expected one active GRIDRES definition in {path}; found {replacements}.")
path.write_text(updated)
if not re.search(r"(?m)^\s*#define\s+GRIDRES\s+0\.15(?:\s|$)", updated):
    raise SystemExit(f"Failed to set GRIDRES to 0.15 in {path}.")
PY

make -C "$source_dir/voro++/src" -j"$build_jobs"
make -C "$source_dir" clean
make -C "$source_dir" -j"$build_jobs" network

[[ -x "$source_dir/network" ]] || { echo "Zeo++ build did not produce network." >&2; exit 1; }
echo "Built Zeo++ with GRIDRES=0.15: $source_dir/network"
echo "Set ZEO_NETWORK=$source_dir/network before submitting preprocessing jobs."
