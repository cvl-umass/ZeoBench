from monai.networks.nets import ViT as ViT_3D
import torch

class ViTMonai3D(ViT_3D):
    """Wrapper for MONAI ViT that handles tuple output and pooling options."""
    def __init__(self, use_cls_token=False, **kwargs):
        self.use_cls_token = use_cls_token
        kwargs.pop('kwargs', None)
        
        # Tell MONAI to initialize the [CLS] token logic
        kwargs["classification"] = use_cls_token
        
        super().__init__(**kwargs)
        self.num_features = kwargs["hidden_size"]

        # THE FIX: Rip out MONAI's built-in 2-class head
        # so it returns the raw 384-dim [CLS] token directly.
        if self.use_cls_token:
            self.classification_head = torch.nn.Identity()

    def forward(self, x):
        x, _ = super().forward(x) 
        
        # If classification=False, x is (Batch, Patches, Dim). We do Global Average Pooling.
        # If classification=True, MONAI natively extracts x[:, 0] and passes it to 
        # classification_head. Because we set that to Identity, x is exactly (Batch, Dim).
        if x.dim() == 3:
            x = x.mean(dim=1)
            
        return x

def _create_vit_monai(default_cfg, kwargs):
    """Helper to merge defaults with config overrides."""
    cfg = default_cfg.copy()
    cfg.update(kwargs)
    
    # Extract custom use_cls_token parameter if it exists (default False)
    use_cls_token = cfg.pop("use_cls_token", False)
    
    return ViTMonai3D(use_cls_token=use_cls_token, **cfg)


def vit_base_patch16_monai_3d(method, **kwargs):
    """Factory function for MONAI's 3D ViT-Base."""
    defaults = {
        "in_channels": 1,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "classification": False,
    }
    return _create_vit_monai(defaults, kwargs)

def vit_large_patch16_monai_3d(method, **kwargs):
    """Factory function for MONAI's 3D ViT-Large."""
    defaults = {
        "in_channels": 1,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
        "hidden_size": 1024,
        "mlp_dim": 4096,
        "num_layers": 24,
        "num_heads": 16,
        "classification": False,
    }
    return _create_vit_monai(defaults, kwargs)

def vit_small_patch16_monai_3d(method, **kwargs):
    """Factory function for MONAI's 3D ViT-Small."""
    defaults = {
        "in_channels": 1,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
        "hidden_size": 384,
        "mlp_dim": 1536,
        "num_layers": 12,
        "num_heads": 6,
        "classification": False,
    }
    return _create_vit_monai(defaults, kwargs)

def vit_tiny_patch16_monai_3d(method, **kwargs):
    """Factory function for MONAI's 3D ViT-Tiny."""
    defaults = {
        "in_channels": 1,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
        "hidden_size": 192,
        "mlp_dim": 768,
        "num_layers": 12,
        "num_heads": 3,
        "classification": False,
    }
    return _create_vit_monai(defaults, kwargs)