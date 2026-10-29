import sys
from pathlib import Path

import torch.nn as nn

# 3DINO is kept alongside code-release in this workspace. Resolve that checkout
# relative to this file instead of relying on the old absolute cluster path.
CODE_RELEASE_ROOT = Path(__file__).resolve().parents[3]
THREE_DINO_ROOT = CODE_RELEASE_ROOT.parent / "3DINO-main"
if (THREE_DINO_ROOT / "dinov2").is_dir():
    sys.path.insert(0, str(THREE_DINO_ROOT))

from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config_3d

def vit_3dino(pretrained_weights="", config_file='train/vit3d_highres', **kwargs):
    cfg = load_and_merge_config_3d(config_file)
    model = build_model_for_eval(cfg, pretrained_weights)
    model.fc = nn.Identity() 
    return model
