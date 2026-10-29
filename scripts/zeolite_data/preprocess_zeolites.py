#!/usr/bin/env python3
"""Generate repository-compatible 0.15-Angstrom Zeo++ distance grids."""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


GRID_RESOLUTION_ANGSTROM = 0.15
DATABASES = {"IZASC", "PCOD"}


@dataclass(frozen=True)
class Structure:
    database: str
    cif_path: Path

    @property
    def name(self) -> str:
        return self.cif_path.stem

    @property
    def selector(self) -> str:
        return f"{self.database}/{self.name}"


def _database_for(path: Path, root: Path) -> str:
    try:
        relative_parts = path.resolve().relative_to(root.resolve()).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]
    for part in reversed(relative_parts):
        database = part.upper()
        if database in DATABASES:
            return database
    if root.name.upper() in DATABASES:
        return root.name.upper()
    raise ValueError(f"Cannot determine IZASC/PCOD from CIF path: {path}")


def _read_selectors(args: argparse.Namespace) -> set[str] | None:
    selectors: list[str] = list(args.molecules or [])
    if args.molecules_file:
        with args.molecules_file.open(encoding="utf-8") as stream:
            for line in stream:
                value = line.split("#", 1)[0].strip()
                if value:
                    selectors.append(value)
    if not selectors:
        return None

    normalized: set[str] = set()
    for value in selectors:
        value = value.strip().replace("\\", "/")
        if "/" in value:
            database, name = value.rsplit("/", 1)
            database = database.upper()
            if database not in DATABASES or not name:
                raise ValueError(f"Invalid molecule selector {value!r}; use DATABASE/ID or ID.")
            normalized.add(f"{database}/{Path(name).stem}")
        else:
            normalized.add(Path(value).stem)
    return normalized


def _read_file_list(path: Path, root: Path) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            cif = Path(value)
            if not cif.is_absolute():
                cif = root / cif
            cif = cif.resolve()
            if cif.suffix.lower() != ".cif":
                raise ValueError(f"{path}:{line_number}: expected a .cif path, got {value!r}")
            if not cif.is_file():
                raise ValueError(f"{path}:{line_number}: CIF does not exist: {cif}")
            if cif not in seen:
                seen.add(cif)
                result.append(cif)
    return result


def _structures(args: argparse.Namespace) -> tuple[list[Structure], set[str] | None]:
    root = args.cif_root.resolve()
    selectors = _read_selectors(args)
    if args.file_list:
        cifs = _read_file_list(args.file_list, root)
    elif selectors:
        # The Zenodo/reference layout has database directories directly under CIFs.
        # This fast path avoids listing the full collection for a small molecule list.
        requested_names = {item.split("/", 1)[-1] for item in selectors}
        databases = {args.database} if args.database else DATABASES
        cifs = []
        for database in sorted(databases):
            search_root = root / database if root.name.upper() not in DATABASES else root
            if search_root.is_dir():
                for name in requested_names:
                    direct = search_root / f"{name}.cif"
                    if direct.is_file():
                        cifs.append(direct.resolve())
    else:
        cifs = sorted(path.resolve() for path in root.rglob("*.cif"))

    records: list[Structure] = []
    found: set[str] = set()
    for cif in cifs:
        database = _database_for(cif, root)
        if args.database and database != args.database:
            continue
        record = Structure(database, cif)
        if selectors and record.selector not in selectors and record.name not in selectors:
            continue
        records.append(record)
        found.add(record.selector)

    if selectors:
        missing = sorted(
            selector
            for selector in selectors
            if selector not in found and selector.split("/", 1)[-1] not in found
        )
        if missing:
            raise ValueError("Requested CIF(s) not found: " + ", ".join(missing))
    records.sort(key=lambda item: item.selector)

    if args.shard_index is not None:
        records = [item for index, item in enumerate(records) if index % args.num_shards == args.shard_index]
    return records, selectors


def _parse_cube(cube_path: Path) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Read a Zeo++ Gaussian cube as x/y/z ordered data and return its origin."""
    with cube_path.open("r", encoding="ascii", errors="replace") as stream:
        first = stream.readline()
        second = stream.readline()
        atom_line = stream.readline().split()
        if not first or not second or len(atom_line) < 4:
            raise ValueError(f"Not a valid Gaussian cube: {cube_path}")
        atom_count = int(atom_line[0])
        origin = np.asarray([float(value) for value in atom_line[1:4]], dtype=np.float64)
        axes: list[np.ndarray] = []
        dimensions: list[int] = []
        for axis in range(3):
            fields = stream.readline().split()
            if len(fields) < 4:
                raise ValueError(f"Incomplete cube axis header in {cube_path}")
            count = int(fields[0])
            if count <= 0:
                raise ValueError(f"Expected Angstrom cube counts; got {count} on axis {axis}")
            dimensions.append(count)
            axes.append(np.asarray([float(value) for value in fields[1:4]], dtype=np.float64))

        if atom_count < 0:
            raise ValueError("Orbital-style Gaussian cube files are not supported.")
        for _ in range(atom_count):
            if not stream.readline():
                raise ValueError(f"Cube ended inside atom records: {cube_path}")

        first_data_line = stream.readline()
        # Zeo++ writes a one-value / one-dataset marker after the atom records.
        # It is not part of the distance array.
        if first_data_line.split() != ["1", "1"]:
            data_lines = itertools.chain((first_data_line,), stream)
        else:
            data_lines = stream

        values = np.empty(int(np.prod(dimensions)), dtype=np.float32)
        offset = 0
        for line in data_lines:
            if not line.strip():
                continue
            block = np.fromstring(line, dtype=np.float64, sep=" ")
            end = offset + block.size
            if end > values.size:
                raise ValueError(f"Too many grid values in {cube_path}")
            values[offset:end] = block
            offset = end
        if offset != values.size:
            raise ValueError(f"Expected {values.size} grid values, read {offset} from {cube_path}")

    grid = values.reshape(tuple(dimensions), order="C")
    return grid, (axes[0], axes[1], axes[2]), origin


def _valid_hdf5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as data:
            grid = data["distancegrids"]
            vectors_valid = all(
                name in grid.attrs
                and np.asarray(grid.attrs[name]).shape == (3,)
                and GRID_RESOLUTION_ANGSTROM * 0.75
                <= float(np.linalg.norm(grid.attrs[name]))
                <= GRID_RESOLUTION_ANGSTROM + 1e-5
                for name in ("xvec", "yvec", "zvec")
            )
            return (
                grid.ndim == 3
                and grid.dtype == np.dtype(np.float32)
                and vectors_valid
            )
    except (OSError, KeyError, ValueError):
        return False


def _process_one(structure: Structure, args: argparse.Namespace, network: Path) -> None:
    target_dir = args.out_root / "distance-grids-h5" / structure.database
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{structure.name}.h5py"
    if target.is_file() and not args.force and _valid_hdf5(target):
        print(f"SKIP {structure.selector}: already has a valid {target.name}", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix=f"zeo-{structure.name}-", dir=args.temp_root) as work_name:
        work = Path(work_name)
        # Keep Zeo++'s default-named outputs in the temporary work directory,
        # even when the input CIF lives in a read-only reference tree.
        local_cif = work / structure.cif_path.name
        shutil.copy2(structure.cif_path, local_cif)
        command = [str(network), "-gridG", local_cif.name]
        completed = subprocess.run(
            command,
            cwd=work,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode:
            print(completed.stdout, file=sys.stderr)
            raise RuntimeError(f"Zeo++ failed for {structure.selector} (exit {completed.returncode}).")
        cubes = sorted(work.glob("*.cube"))
        if len(cubes) != 1:
            raise RuntimeError(
                f"Expected one .cube file for {structure.selector}; found {len(cubes)}. "
                "Check the installed Zeo++ build and -gridG support."
            )
        grid, vectors, origin = _parse_cube(cubes[0])
        if not np.allclose(origin, 0.0, atol=1e-8, rtol=0.0):
            raise RuntimeError(
                f"{structure.selector} has cube origin {origin.tolist()}; this loader format assumes origin (0, 0, 0)."
            )

    fd, temporary_name = tempfile.mkstemp(prefix=f".{structure.name}.", suffix=".h5py.tmp", dir=target_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as output:
            dataset = output.create_dataset("distancegrids", data=grid, dtype=np.float32)
            dataset.attrs["xvec"] = vectors[0]
            dataset.attrs["yvec"] = vectors[1]
            dataset.attrs["zvec"] = vectors[2]
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    steps = [float(np.linalg.norm(vector)) for vector in vectors]
    print(
        f"DONE {structure.selector}: shape={tuple(grid.shape)} "
        f"step_A={[round(value, 6) for value in steps]} -> {target}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-root", required=True, type=Path, help="Directory containing IZASC/ and PCOD/ CIF subdirectories")
    parser.add_argument("--out-root", required=True, type=Path, help="Dataset root; output is written under distance-grids-h5/")
    parser.add_argument("--network", type=Path, default=os.environ.get("ZEO_NETWORK"), help="Zeo++ network binary (or set ZEO_NETWORK)")
    parser.add_argument("--database", choices=sorted(DATABASES), help="Limit discovery to IZASC or PCOD")
    parser.add_argument("--molecules", nargs="+", help="IDs or selectors such as IZASC/ABW-0")
    parser.add_argument("--molecules-file", type=Path, help="Text file with one ID or DATABASE/ID per line")
    parser.add_argument("--file-list", type=Path, help="Sorted manifest of CIF paths; make once for efficient full-data sharding")
    parser.add_argument("--shard-index", type=int, help="Zero-based shard index; use with --num-shards")
    parser.add_argument("--num-shards", type=int, help="Total number of deterministic shards")
    parser.add_argument("--temp-root", type=Path, default=None, help="Temporary space for Zeo++ cube files (defaults to system temp)")
    parser.add_argument("--force", action="store_true", help="Reprocess and atomically replace existing outputs")
    parser.add_argument("--allow-non-slurm", action="store_true", help="Allow processing outside Slurm (for a workstation only)")
    args = parser.parse_args()

    try:
        if not os.environ.get("SLURM_JOB_ID") and not args.allow_non_slurm:
            raise ValueError(
                "Refusing to preprocess outside a Slurm allocation. Submit with sbatch/srun; "
                "--allow-non-slurm is intended for a workstation, not an HPC login node."
            )
        if (args.shard_index is None) != (args.num_shards is None):
            raise ValueError("Use --shard-index and --num-shards together.")
        if args.num_shards is not None and (args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards):
            raise ValueError("Shard index must satisfy 0 <= index < number of shards.")
        if not args.cif_root.is_dir():
            raise ValueError(f"CIF root is not a directory: {args.cif_root}")
        if args.temp_root:
            args.temp_root.mkdir(parents=True, exist_ok=True)
        network = Path(args.network).expanduser().resolve() if args.network else None
        if network is None or not network.is_file() or not os.access(network, os.X_OK):
            raise ValueError("Set --network or ZEO_NETWORK to an executable Zeo++ network binary.")

        structures, selectors = _structures(args)
        if not structures:
            raise ValueError("No CIF structures selected.")
        print(
            f"Processing {len(structures)} structure(s) at {GRID_RESOLUTION_ANGSTROM:.2f} Å; "
            f"output root: {args.out_root.resolve()}",
            flush=True,
        )
        if args.shard_index is not None:
            print(f"Shard {args.shard_index}/{args.num_shards}", flush=True)
        for structure in structures:
            _process_one(structure, args, network)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"preprocess_zeolites.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
