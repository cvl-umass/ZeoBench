import torch
import torch.nn as nn
# Import the inner SwinTransformer class from MONAI
from monai.networks.nets.swin_unetr import SwinTransformer

class SwinMonai3D(SwinTransformer):
    """Wrapper for MONAI 3D SwinTransformer (from SwinUNETR) for classification/SSL."""
    def __init__(self, **kwargs):
        kwargs.pop('kwargs', None)
        super().__init__(**kwargs)
        
        actual_embed_dim = kwargs.get("embed_dim", 48)
        self.num_features = actual_embed_dim * 16
        
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        outs = super().forward(x, normalize=True)
        feat = outs[-1] 
        feat = self.pool(feat)
        return feat.flatten(1)

def _create_swin_monai(default_cfg, kwargs):
    cfg = default_cfg.copy()
    cfg.update(kwargs)
    if "window_size" in cfg and isinstance(cfg["window_size"], int):
        cfg["window_size"] = (cfg["window_size"],) * 3
        
    if "patch_size" in cfg and isinstance(cfg["patch_size"], int):
        cfg["patch_size"] = (cfg["patch_size"],) * 3
    return SwinMonai3D(**cfg)

def swin_tiny_monai_3d(method, **kwargs):
    """Factory function for 3D Swin-Tiny."""
    defaults = {
        "in_chans": 1,
        "spatial_dims": 3,
        "embed_dim": 96,
        "window_size": (7, 7, 7),
        "patch_size": (2, 2, 2),
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    }
    return _create_swin_monai(defaults, kwargs)

def swin_small_monai_3d(method, **kwargs):
    """Factory function for 3D Swin-Small."""
    defaults = {
        "in_chans": 1,
        "spatial_dims": 3,
        "embed_dim": 96,
        "window_size": (7, 7, 7),
        "patch_size": (2, 2, 2),
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
    }
    return _create_swin_monai(defaults, kwargs)

def swin_base_monai_3d(method, **kwargs):
    """Factory function for 3D Swin-Base."""
    defaults = {
        "in_chans": 1,
        "spatial_dims": 3,
        "embed_dim": 128,
        "window_size": (7, 7, 7),
        "patch_size": (2, 2, 2),
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    }
    return _create_swin_monai(defaults, kwargs)

def swin_large_monai_3d(method, **kwargs):
    """Factory function for 3D Swin-Large."""
    defaults = {
        "in_chans": 1,
        "spatial_dims": 3,
        "embed_dim": 192,
        "window_size": (7, 7, 7),
        "patch_size": (2, 2, 2),
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    }
    return _create_swin_monai(defaults, kwargs)
