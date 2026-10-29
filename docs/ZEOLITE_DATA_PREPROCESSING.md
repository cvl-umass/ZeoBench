# Zeolite download and distance grids

This workflow downloads the CIF files from the [Zenodo record 23736688](https://zenodo.org/records/23736688), builds [Zeo++ 0.3](https://www.zeoplusplus.org/download.html), and writes 0.15 Å distance grids in the format read by `solo/data/zeolites_dg.py`. That loader expects one `distance-grids-h5/{IZASC,PCOD}/<ID>.h5py` file per CIF, with a `distancegrids` float32 array and `xvec`, `yvec`, and `zvec` attributes.

Use Python with NumPy and h5py installed. The preprocessor requires a Slurm allocation by default; `--allow-non-slurm` is only for a workstation and should not be used on a Unity login node.

## Download and build Zeo++

```bash
DATA_ROOT=/path/to/zeolite-data
python scripts/zeolite_data/download_zenodo.py --dest "$DATA_ROOT"
BUILD_JOBS=2 scripts/zeolite_data/setup_zeopp.sh "$HOME/opt/zeopp-0.3-grid015"
export ZEO_NETWORK="$HOME/opt/zeopp-0.3-grid015/zeo++-0.3/network"
```

Downloaded record files are kept under `downloads/`; supported ZIP and tar archives are unpacked under `structures/`. Point `CIF_ROOT` at the directory containing `IZASC/` and `PCOD/` (often `"$DATA_ROOT/structures/CIFs"`). If Zenodo requires access, set `ZENODO_ACCESS_TOKEN` in your shell environment; never put its value in the repository.

Downloading and unpacking the full record can be high-volume I/O. Use `srun` or `sbatch` for a large transfer, with the destination on appropriate work storage.

The setup script follows the Zeo++ [distance-grid example](https://www.zeoplusplus.org/examples.html#grid), sets `GRIDRES` in `grid.cc` to `0.15`, then builds the bundled Voro++ library and `network` executable. The preprocessor uses `network -gridG` and converts its Gaussian cube output to the repository HDF5 layout.

## Preprocess selected structures or shards

Run the preprocessor inside a CPU Slurm job or array, not directly on a login node. It skips valid output files on reruns and writes each new HDF5 file atomically.

```bash
python scripts/zeolite_data/preprocess_zeolites.py \
  --cif-root "$CIF_ROOT" --out-root "$DATA_ROOT/processed" \
  --network "$ZEO_NETWORK" --molecules IZASC/ABW-0 PCOD/8000001
```

For a full run, build one sorted CIF manifest and submit an array. Each task handles a deterministic shard; choose task resources and array size from a small Slurm pilot.

```bash
find "$CIF_ROOT" -type f -iname '*.cif' | LC_ALL=C sort > "$DATA_ROOT/cifs.txt"
python scripts/zeolite_data/preprocess_zeolites.py \
  --cif-root "$CIF_ROOT" --file-list "$DATA_ROOT/cifs.txt" \
  --out-root "$DATA_ROOT/processed" --network "$ZEO_NETWORK" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --num-shards 32
```

Alternatively, use `--molecules-file` with one `ID` or `DATABASE/ID` per line. Pass `--force` to deliberately regenerate existing grids.

## Small reference check

The CPU-only batch script processes three fixed, seeded sample molecules and compares grid dimensions, axis vectors, and values against an existing processed reference dataset. Set `ZEOLITE_REFERENCE_ROOT` to its root before submission:

```bash
export ZEO_NETWORK="$HOME/opt/zeopp-0.3-grid015/zeo++-0.3/network"
export PYTHON=/path/to/python-with-numpy-and-h5py
export ZEOLITE_REFERENCE_ROOT=/path/to/ML-Zeolites
sbatch --export=ALL scripts/zeolite_data/test_preprocess_cpu.sbatch
```

See [the test report](ZEOLITE_DATA_PREPROCESSING_TEST.md) for the selected molecules, reference metadata, and test status.
