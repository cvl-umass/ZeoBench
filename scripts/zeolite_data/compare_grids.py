#!/usr/bin/env python3
"""Compare generated HDF5 grids with an existing Zeolite dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def _selectors(args: argparse.Namespace) -> list[tuple[str | None, str]]:
    values = list(args.molecules or [])
    if args.molecules_file:
        with args.molecules_file.open(encoding="utf-8") as stream:
            values.extend(line.split("#", 1)[0].strip() for line in stream)
    result: list[tuple[str | None, str]] = []
    seen = set()
    for value in values:
        value = value.strip().replace("\\", "/")
        if not value:
            continue
        if "/" in value:
            database, name = value.rsplit("/", 1)
            database = database.upper()
        else:
            database, name = args.database, value
        key = (database, Path(name).stem)
        if key not in seen:
            seen.add(key)
            result.append(key)
    if not result:
        raise ValueError("Provide --molecules or --molecules-file.")
    return result


def _read_grid(path: Path):
    with h5py.File(path, "r") as source:
        values = source["distancegrids"][:]
        vectors = tuple(np.asarray(source["distancegrids"].attrs[name], dtype=np.float64) for name in ("xvec", "yvec", "zvec"))
    return values, vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True, type=Path, help="Root written by preprocess_zeolites.py")
    parser.add_argument("--reference-root", required=True, type=Path, help="Read-only dataset root containing distance-grids-h5/")
    parser.add_argument("--molecules", nargs="+", help="IDs or selectors such as PCOD/8163091")
    parser.add_argument("--molecules-file", type=Path, help="Text file with one ID or DATABASE/ID per line")
    parser.add_argument("--database", choices=("IZASC", "PCOD"), help="Database for bare IDs")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for axis vectors and distance values")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for distance values")
    args = parser.parse_args()

    try:
        failed = False
        for database, name in _selectors(args):
            databases = [database] if database else ["IZASC", "PCOD"]
            candidates = []
            for db in databases:
                generated = args.generated_root / "distance-grids-h5" / db / f"{name}.h5py"
                reference = args.reference_root / "distance-grids-h5" / db / f"{name}.h5py"
                if generated.is_file() or reference.is_file():
                    candidates.append((db, generated, reference))
            if not candidates:
                print(f"FAIL {name}: not found in either dataset root", flush=True)
                failed = True
                continue
            if len(candidates) > 1 and database is None:
                print(f"FAIL {name}: ID exists in both databases; qualify it as DATABASE/ID", flush=True)
                failed = True
                continue

            for db, generated_path, reference_path in candidates:
                label = f"{db}/{name}"
                if not generated_path.is_file() or not reference_path.is_file():
                    print(f"FAIL {label}: generated={generated_path.is_file()} reference={reference_path.is_file()}", flush=True)
                    failed = True
                    continue

                generated, generated_vectors = _read_grid(generated_path)
                reference, reference_vectors = _read_grid(reference_path)
                shape_ok = generated.shape == reference.shape
                vector_diff = max(
                    float(np.max(np.abs(left - right)))
                    for left, right in zip(generated_vectors, reference_vectors)
                )
                vectors_ok = all(
                    left.shape == (3,) and right.shape == (3,)
                    and np.allclose(left, right, atol=args.atol, rtol=0.0)
                    for left, right in zip(generated_vectors, reference_vectors)
                )
                shape_status = "PASS" if shape_ok else "FAIL"
                vector_status = "PASS" if vectors_ok else "FAIL"
                print(f"{label}: layout={shape_status}, axes/alignment={vector_status}")
                print(f"  generated shape={generated.shape}, reference shape={reference.shape}")
                print(f"  max |axis-vector delta|={vector_diff:.8g} Å")
                if not shape_ok:
                    failed = True
                    continue

                difference = generated.astype(np.float64) - reference.astype(np.float64)
                abs_difference = np.abs(difference)
                max_abs = float(abs_difference.max(initial=0.0))
                mae = float(abs_difference.mean())
                rmse = float(np.sqrt(np.mean(difference * difference)))
                exact_percent = 100.0 * float(np.count_nonzero(generated == reference)) / generated.size
                values_ok = bool(np.allclose(generated, reference, atol=args.atol, rtol=args.rtol, equal_nan=True))
                print(
                    f"  values={'PASS' if values_ok else 'FAIL'}; max_abs={max_abs:.8g} Å, "
                    f"MAE={mae:.8g} Å, RMSE={rmse:.8g} Å, exact={exact_percent:.3f}%"
                )
                if not vectors_ok or not values_ok:
                    failed = True
        return 1 if failed else 0
    except (OSError, KeyError, ValueError) as exc:
        print(f"compare_grids.py: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
