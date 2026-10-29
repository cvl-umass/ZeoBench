import os
import csv
import torch
import torch.nn as nn
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from solo.methods.base import BaseMethod

class ZeoliteFeatureCacheDataset(Dataset):
    def __init__(self, data_dir, valid_names, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.distance_folder = 'distance-grids-h5'
        
        # Map zeolite names to their respective DB set
        self.name_to_set = {}
        csv_path = os.path.join(data_dir, 'C18-adsorption', 'each-zeolite-info.csv')
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for line in reader:
                db_set = "IZASC" if line[1] == "IZA" else "PCOD"
                self.name_to_set[line[0]] = db_set
                
        # Filter for valid files that actually exist
        self.valid_names = []
        for n in valid_names:
            if n in self.name_to_set:
                db_set = self.name_to_set[n]
                if os.path.exists(os.path.join(self.data_dir, self.distance_folder, db_set, n + '.h5py')):
                    self.valid_names.append(n)

    def __len__(self):
        return len(self.valid_names)

    def __getitem__(self, idx):
        name = self.valid_names[idx]
        db_set = self.name_to_set[name]
        distance_name = os.path.join(self.data_dir, self.distance_folder, db_set, name + '.h5py')
        
        with h5py.File(distance_name, 'r') as f:
            grid = torch.Tensor(np.array(f['distancegrids'])).unsqueeze(0)
            _, nx, ny, nz = grid.shape
            bounds = torch.Tensor([nx, ny, nz])
            xvec = f['distancegrids'].attrs['xvec']
            yvec = f['distancegrids'].attrs['yvec']
            zvec = f['distancegrids'].attrs['zvec']
        
        raw_data = (grid, xvec, yvec, zvec, bounds, nx, ny, nz)
        if self.transform:
            grid = self.transform(raw_data)
        return name, grid


def get_valid_zeolites(data_dir):
    """Finds all zeolites that have at least one non-zero label across all molecules."""
    csv_path = os.path.join(data_dir, 'C18-adsorption', 'each-zeolite-info.csv')
    valid_names = set()
    mol_to_idx = {'C18': 2, 'C24': 3, 'C30': 4, '2C17': 5, '4C17': 6, '22C16': 7}
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader) 
        for line in reader:
            # valid_names.add(line[0])
            for idx in mol_to_idx.values():
                if line[idx] != '0':
                    valid_names.add(line[0])
                    break
    return valid_names

@hydra.main(version_base="1.2", config_path="configs", config_name="main_linear")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(cfg)
    
    # 1. Setup Paths
    # Pull the path directly from the dataset config instead of argparse
    data_dir = cfg.data.get("val_path", "/path/to/dataset/zeolites")
    valid_names = get_valid_zeolites(data_dir)
    print(f"Found {len(valid_names)} valid zeolites.")

    # Hydra automatically creates a unique runtime directory for us!
    job_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Saving features to: {job_dir}")

    # 2. Setup TTA and Transforms
    num_views = cfg.get("num_tta_views", 10)
    if num_views > 1:
        T_val = hydra.utils.instantiate(cfg.train_transform)
        print(f"Using train_transform for TTA with {num_views} views.")
    else:
        T_val = hydra.utils.instantiate(cfg.val_transform)

    dataset = ZeoliteFeatureCacheDataset(data_dir, valid_names, transform=T_val)

    # Use standard batch_size from optimizer config if not explicitly overridden
    batch_size = cfg.get("batch_size", cfg.optimizer.batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    # 3. Instantiate Backbone
    if "_target_" in cfg.backbone:
        with open_dict(cfg.backbone):
            # Extract the kwargs dictionary, defaulting to an empty dict
            backbone_kwargs = cfg.backbone.pop("kwargs", {})
            
            if "torchvision" not in cfg.backbone["_target_"]:
                for key, value in backbone_kwargs.items():
                    cfg.backbone[key] = value
            
    # Hydra will now pass the flattened keys natively
    backbone = hydra.utils.instantiate(cfg.backbone)
    
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    if hasattr(backbone, "heads"):
        backbone.heads.head = nn.Identity()
    
    # 4. Handle Checkpoint vs. ImageNet
    ckpt_val = cfg.get("pretrained_feature_extractor", "")
    if ckpt_val == "imagenet":
        print("Using ImageNet weights, bypassing checkpoint loading.")
    else:
        # Load specific checkpoint if provided
        state = torch.load(ckpt_val, map_location="cpu", weights_only=False)["state_dict"]
        new_state = {k.replace("encoder", "backbone").replace("backbone.", ""): v for k, v in state.items()}
        backbone.load_state_dict(new_state, strict=False)
        print(f"Loaded weights from {ckpt_val}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()

    # 5. Extract Features & TTA Averaging
    cache_dict = {}
    with torch.no_grad():
        for view_idx in range(num_views):
            if num_views > 1: print(f"Extracting TTA View {view_idx+1}/{num_views}")
            
            for names, X in tqdm(dataloader, desc="Extracting"):
                X = X.to(device)
                if X.ndim == 5: X = X.to(memory_format=torch.channels_last_3d)
                elif X.ndim == 4: X = X.to(memory_format=torch.channels_last)

                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    feats = backbone(X)
                
                feats = feats.cpu().to(torch.float32)
                
                for name, feat in zip(names, feats):
                    if name not in cache_dict: 
                        cache_dict[name] = feat.clone()
                    else: 
                        cache_dict[name] += feat

    for name in cache_dict:
        cache_dict[name] = (cache_dict[name] / num_views).to(torch.float16)
    
    out_file = os.path.join(job_dir, "offline_features.pt")
    torch.save(cache_dict, out_file)
    print(f"Successfully cached features to {out_file}")

if __name__ == "__main__":
    main()
