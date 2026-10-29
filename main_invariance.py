import os
import csv
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate

from solo.methods.linear import LinearModel
from solo.methods.base import BaseMethod
from solo.data.zeolites_dg import ZeoDataset
from solo.data.custom_transforms import ZeoAugmentation, QuantileNormalize3DINO

class DeterministicMultiAugmentTransform:
    """
    Applies N deterministic augmentations to a data item by pre-generating 
    or loading fixed translation vectors and rotation matrices.
    **VECTORIZED for massive CPU speedups.**
    """
    def __init__(self, base_transform, num_augs=100, seed=42, mode="both", aug_dir="./subsets/augs", use_3dino_norm=False):
        self.base_transform = base_transform
        self.num_augs = num_augs
        self.mode = mode
        self.use_3dino_norm = use_3dino_norm
        
        if self.use_3dino_norm:
            self.dino_norm = QuantileNormalize3DINO()
        
        os.makedirs(aug_dir, exist_ok=True)
        aug_file = os.path.join(aug_dir, f"seed_{seed}_augs_{num_augs}.pt")
        
        if os.path.exists(aug_file):
            print(f"🔄 Loading deterministic augmentations from {aug_file}")
            augs = torch.load(aug_file)
            self.u_trans = augs['u_trans']
            self.rot_mats = augs['rot_mats']
        else:
            print(f"⚙️ Generating and saving deterministic augmentations to {aug_file}")
            torch.manual_seed(seed)
            self.u_trans = torch.rand(num_augs, 3)
            
            self.rot_mats = []
            for _ in range(num_augs):
                p = torch.randn(3, 3)
                rotation_matrix, _ = torch.linalg.qr(p)
                rand_int = torch.randint(0, 2, (1,))
                rotation_matrix = (2 * rand_int - 1) * rotation_matrix
                rotation_matrix = torch.transpose(rotation_matrix, 0, 1)
                self.rot_mats.append(rotation_matrix)
                
            self.rot_mats = torch.stack(self.rot_mats)
            torch.save({'u_trans': self.u_trans, 'rot_mats': self.rot_mats}, aug_file)

    def __call__(self, data_item):
        grid, xvec, yvec, zvec, bounds, nx, ny, nz = data_item
        
        grid_res_val = self.base_transform.base_grid_resolution
        A_inv = self.base_transform._get_Ainv_matrix(xvec, yvec, zvec, grid_res_val, grid_res_val, grid_res_val)
        
        # 1. Base coordinates (Expand to batch dimension: [num_augs, H*W*D, 3])
        coord = self.base_transform.coord.unsqueeze(0).expand(self.num_augs, -1, -1).clone()

        # 2. Vectorized Translation
        if self.mode in ["both", "translate"]:
            # Shape u_trans: [num_augs, 3] -> [num_augs, 1, 3] to broadcast across H*W*D
            trans_vec = (self.u_trans * bounds).unsqueeze(1)
            coord += trans_vec
            
        # 3. Vectorized Rotation
        if self.mode in ["both", "rotate"]:
            # Batch Matrix Multiplication: [num_augs, H*W*D, 3] @ [num_augs, 3, 3]
            coord = torch.bmm(coord, self.rot_mats)

        # 4. Map coordinates to grid (Vectorized version of ZeoAugmentation logic)
        bounds_tensor = torch.Tensor([nx, ny, nz])
        
        # Apply A_inv (broadcast over batch)
        grid3d = torch.remainder(torch.matmul(coord, A_inv), bounds_tensor)
        
        # Scale to [-1, 1]
        scalar = torch.diag(2 / bounds_tensor)
        grid3d = torch.add(torch.matmul(grid3d, scalar), -1)

        # Reshape to 5D grid for F.grid_sample: [num_augs, H, W, D, 3]
        H, W, D = self.base_transform.H, self.base_transform.W, self.base_transform.D
        transf_grid = grid3d.reshape(self.num_augs, H, W, D, 3)

        # Expand original 3D volume to mimic the batch dimension: [num_augs, 1, nz, ny, nx]
        grid_vol = grid.permute(0, 3, 2, 1).unsqueeze(0) 
        grid_vol = grid_vol.expand(self.num_augs, -1, -1, -1, -1) 

        # Single batched grid sample for all 100 augmentations!
        grid_res = torch.nn.functional.grid_sample(grid_vol, transf_grid, align_corners=True) 

        # Apply 3DINO normalization sequentially across the batch dimension
        if getattr(self, 'use_3dino_norm', False):
            normalized_grids = []
            for i in range(self.num_augs):
                normalized_grids.append(self.dino_norm(grid_res[i]))
            grid_res = torch.stack(normalized_grids)

        return grid_res


def main():
    parser = argparse.ArgumentParser(description="Evaluate translational/rotational invariance deterministically.")
    parser.add_argument("--job_dir", type=str, required=True, help="Path to job directory containing args.json and last.ckpt")
    parser.add_argument("--num_augs", type=int, default=100, help="Number of augmentations per datapoint")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--mode", type=str, choices=["both", "rotate", "translate"], default="both", help="Flag to isolate augmentation types.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Mini-batch size to prevent GPU OOM during inference.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic augmentations.")
    parser.add_argument("--aug_dir", type=str, default="../subset/augs", help="Directory to save/load augmentation parameters.")
    parser.add_argument("--out_csv", type=str, default="invariance_results.csv", help="Output filename.")

    parser.add_argument("--crop_size", type=int, default=None, help="Override crop size (required for nested configs like 3DINO)")
    parser.add_argument("--grid_resolution", type=float, default=None, help="Override grid resolution (required for nested configs like 3DINO)")
    parser.add_argument("--use_3dino_norm", action="store_true", help="Apply QuantileNormalize3DINO to augmentations.")
    args = parser.parse_args()

    # 1. Load Original Configuration
    config_path = os.path.join(args.job_dir, "args.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find args.json in {args.job_dir}.")
    
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = OmegaConf.create(cfg_dict)
    cfg.cache_features = False
    print(cfg)

    # 2. Re-instantiate Backbone and LinearModel
    if "_target_" in cfg.backbone:
        backbone = instantiate(cfg.backbone)
    else:
        backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
        backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)

    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    if hasattr(backbone, "heads"):
        backbone.heads.head = nn.Identity()

    loss_func = nn.MSELoss() if cfg.data.get("task", "classification") == "regression" else nn.CrossEntropyLoss()
    model = LinearModel(backbone, loss_func=loss_func, mixup_func=None, cfg=cfg)

    # 3. Load Model Weights
    ckpt_path = os.path.join(args.job_dir, "last.ckpt")
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Detect if the model was trained on cached offline features (like 3DINO)
    pretrained_ext = str(cfg.get("pretrained_feature_extractor", ""))
    if pretrained_ext.endswith("offline_features.pt"):
        print("Detected offline features cache. Loading ONLY classifier weights to preserve backbone initialization...")
        # Filter out random backbone weights saved in the linear probe checkpoint
        state_dict = {k: v for k, v in state_dict.items() if "classifier" in k}
        model.load_state_dict(state_dict, strict=False)
    else:
        # Standard load for end-to-end models (like scratch.json)
        model.load_state_dict(state_dict)
    model.to(device)
    if not cfg.performance.disable_channel_last:
        if any(isinstance(m, nn.Conv3d) for m in model.modules()):
            model = model.to(memory_format=torch.channels_last_3d)
        else:
            model = model.to(memory_format=torch.channels_last)
    model.eval()

    # 4. Configure Deterministic Augmentations
    val_transform_cfg = cfg.get("val_transform", {})
    
    # Use CLI args if provided, otherwise attempt to parse from config, otherwise default
    crop_size = args.crop_size if args.crop_size is not None else val_transform_cfg.get("crop_size", 100)
    grid_resolution = args.grid_resolution if args.grid_resolution is not None else val_transform_cfg.get("grid_resolution", 0.45)
    
    # Force the base transform to be completely deterministic
    base_transform = ZeoAugmentation(
        crop_size=crop_size,
        grid_resolution=grid_resolution,
        rotate=False,       # Must be False for controlled invariance testing
        translate=False,    # Must be False for controlled invariance testing
        grid_res_sigma=0.0  # Disable any random resolution jitter
    )
    
    multi_transform = DeterministicMultiAugmentTransform(
        base_transform=base_transform,
        num_augs=args.num_augs,
        seed=args.seed,
        mode=args.mode,
        aug_dir=args.aug_dir,
        use_3dino_norm=args.use_3dino_norm
    )

    # 5. Load Dataset
    dataset_kwargs = cfg.data.get("dataset_kwargs", {})
    test_dataset = ZeoDataset(
        root_dir=cfg.data.val_path,
        is_train=False,
        is_val=False,
        transform=multi_transform,
        **dataset_kwargs
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # 6. Evaluation Loop
    results = []
    
    for i, (X_stack, y) in enumerate(tqdm(test_loader, desc=f"Evaluating Invariance (Mode: {args.mode})")):
        molecule_name = test_dataset.files[i]['zeolite']
        X_stack = X_stack.squeeze(0) 
        
        all_preds = []
        
        # Batch inference processing
        for j in range(0, args.num_augs, args.chunk_size):
            X_chunk = X_stack[j:j + args.chunk_size].to(device)
            
            if not cfg.performance.disable_channel_last:
                if X_chunk.ndim == 5:
                    X_chunk = X_chunk.to(memory_format=torch.channels_last_3d)
                elif X_chunk.ndim == 4:
                    X_chunk = X_chunk.to(memory_format=torch.channels_last)
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(X_chunk)["logits"]
                    all_preds.append(logits.float().cpu())
                    
        all_preds = torch.cat(all_preds, dim=0) 
        
        std_per_class = all_preds.std(dim=0)
        mean_std = std_per_class.mean().item()
        
        results.append({
            "zeolite": molecule_name,
            "std": mean_std
        })

    # 7. Write to CSV
    out_filename = args.out_csv.replace(".csv", f"_{args.mode}.csv")
    out_path = os.path.join(args.job_dir, out_filename)
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["zeolite", "std"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Finished processing {len(results)} datapoints.")
    print(f"✅ Results saved to: {out_path}")

if __name__ == "__main__":
    main()