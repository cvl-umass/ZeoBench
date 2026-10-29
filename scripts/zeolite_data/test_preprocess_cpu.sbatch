#!/usr/bin/env bash
# CPU-only correctness check; run with sbatch from the repository root.
#SBATCH --partition=cpu
#SBATCH --job-name=zeolite-grid-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=zeolite-grid-check-%j.out
#SBATCH --error=zeolite-grid-check-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root=$(cd "$SLURM_SUBMIT_DIR" && pwd)
else
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd "$script_dir/../.." && pwd)
fi
script_dir="$repo_root/scripts/zeolite_data"
reference_root=${ZEOLITE_REFERENCE_ROOT:-}
network=${ZEO_NETWORK:-}
python_bin=${PYTHON:-python3}

if [[ -z "$reference_root" ]]; then
  echo "Set ZEOLITE_REFERENCE_ROOT to an existing processed dataset before submission." >&2
  exit 2
fi
if [[ -z "$network" || ! -x "$network" ]]; then
  echo "Set ZEO_NETWORK to the executable Zeo++ network binary before submission." >&2
  exit 2
fi
if [[ ! -d "$reference_root/CIFs" || ! -d "$reference_root/distance-grids-h5" ]]; then
  echo "Reference dataset is missing under $reference_root" >&2
  exit 2
fi
"$python_bin" -c 'import h5py, numpy'

test_root=$(mktemp -d "${TMPDIR:-/tmp}/zeolite-grid-test.XXXXXX")
trap 'rm -rf "$test_root"' EXIT

"$python_bin" "$script_dir/preprocess_zeolites.py" \
  --cif-root "$reference_root/CIFs" \
  --out-root "$test_root/generated" \
  --network "$network" \
  --molecules-file "$script_dir/test_molecules.txt"

"$python_bin" "$script_dir/compare_grids.py" \
  --generated-root "$test_root/generated" \
  --reference-root "$reference_root" \
  --molecules-file "$script_dir/test_molecules.txt"
