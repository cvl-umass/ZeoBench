# Zeolite preprocessing validation

On 2026-10-28, the preprocessing workflow was checked against an existing processed Zeolite dataset using three seeded structures: one IZA molecule and two PCOD molecules. Zeo++ 0.3 generated the distance grids at 0.15 Å resolution. The full dataset was not downloaded or processed.

The CPU Slurm test verified the expected HDF5 layout, float32 `distancegrids` array, axis-vector attributes, grid dimensions, and voxel values. All three grids matched their reference values exactly after float32 storage.

| Molecule | Grid shape (x, y, z) | Axis steps (Å) | Maximum absolute difference |
|---|---:|---|---:|
| `IZASC/AET-0` | `(220, 97, 57)` | `(0.149904, 0.149792, 0.149536)` | `0` Å |
| `PCOD/8163091` | `(166, 72, 103)` | `(0.149238, 0.148062, 0.149097)` | `0` Å |
| `PCOD/8262181` | `(60, 130, 175)` | `(0.149341, 0.149726, 0.149888)` | `0` Å |

This is a small correctness check, not a full-dataset throughput benchmark. See [the preprocessing guide](ZEOLITE_DATA_PREPROCESSING.md) for how to run the check with a reference dataset.
