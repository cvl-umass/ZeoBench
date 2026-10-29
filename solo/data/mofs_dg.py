"""MOF adsorption regression dataset backed by HDF5 distance grids."""

import csv
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MofDataset(Dataset):
    """Load MOF distance grids and one numeric adsorption target.

    Annotation CSVs provide ``mof`` IDs and an ``adsorption`` column. Split
    files contain one MOF ID per line (an optional ``mof``/``zeolite`` header
    is accepted). Distance grids are stored under
    ``<root>/distance-grids-h5/hMOF/<mof>.h5py``.
    """

    sample_id_key = "mof"
    target_key = "adsorption"

    def __init__(
        self,
        root_dir,
        target_column="adsorption",
        annotations_folder="CH4-adsorption",
        annotations_file="hmof_CH4_0.5bar.csv",
        distance_folder="distance-grids-h5",
        train_file="train_set.txt",
        val_file="val_set.txt",
        test_file="test_set.txt",
        custom_train_file=None,
        is_train=True,
        is_val=False,
        transform=None,
        **_unused,
    ):
        self.root_dir = Path(root_dir)
        self.annotations_folder = Path(annotations_folder)
        if not self.annotations_folder.is_absolute():
            self.annotations_folder = self.root_dir / self.annotations_folder
        self.distance_dir = Path(distance_folder)
        if not self.distance_dir.is_absolute():
            self.distance_dir = self.root_dir / self.distance_dir
        self.target_key = target_column
        self.transform = transform

        if is_train:
            split_path = custom_train_file or train_file
        elif is_val:
            split_path = val_file
        else:
            split_path = test_file
        split_path = Path(split_path)
        if not split_path.is_absolute():
            split_path = self.annotations_folder / split_path

        ids = []
        with split_path.open(newline="") as stream:
            for row in csv.reader(stream):
                if not row:
                    continue
                mof_id = row[0].strip()
                if mof_id.lower() in {"mof", "zeolite", "name"}:
                    continue
                if mof_id:
                    ids.append(mof_id)

        requested_ids = set(ids)
        annotation_path = self.annotations_folder / annotations_file
        available = {}
        with annotation_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                mof_id = (row.get("mof") or "").strip()
                if mof_id not in requested_ids:
                    continue
                raw_target = row.get(target_column)
                if raw_target is None or not raw_target.strip():
                    continue
                try:
                    target = float(raw_target)
                except ValueError:
                    continue
                if not np.isfinite(target):
                    continue

                grid_path = self.distance_dir / "hMOF" / f"{mof_id}.h5py"
                if not os.path.exists(str(grid_path)):
                    continue
                available[mof_id] = {
                    self.sample_id_key: mof_id,
                    self.target_key: target,
                    "grid_path": str(grid_path),
                }
        self.files = [available[mof_id] for mof_id in ids if mof_id in available]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        record = self.files[idx]
        with h5py.File(record["grid_path"], "r") as grid_file:
            distance_grid = grid_file["distancegrids"]
            grid = torch.as_tensor(np.asarray(distance_grid), dtype=torch.float32).unsqueeze(0)
            _, nx, ny, nz = grid.shape
            bounds = torch.tensor([nx, ny, nz], dtype=torch.float32)
            xvec = distance_grid.attrs["xvec"]
            yvec = distance_grid.attrs["yvec"]
            zvec = distance_grid.attrs["zvec"]

        raw_data = (grid, xvec, yvec, zvec, bounds, nx, ny, nz)
        if self.transform is not None:
            grid = self.transform(raw_data)
        target = torch.tensor(record[self.target_key], dtype=torch.float32)
        return grid, target
