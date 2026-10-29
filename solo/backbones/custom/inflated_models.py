import math
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

def inflate_conv_weights(conv2d, new_in_channels, scaling_method='linear'):
    out_channels, in_channels, k1, k2 = conv2d.weight.shape
    new_conv = nn.Conv2d(new_in_channels, out_channels, kernel_size=(k1, k2), 
                         stride=conv2d.stride, padding=conv2d.padding, bias=(conv2d.bias is not None))
    with torch.no_grad():
        repeats = int(np.ceil(new_in_channels / 3.0))
        repeated_weight = conv2d.weight.repeat(1, repeats, 1, 1)[:, :new_in_channels, :, :]
        
        if scaling_method == 'linear':
            repeated_weight *= (3.0 / new_in_channels)
        elif scaling_method == 'sqrt':
            repeated_weight *= math.sqrt(3.0 / new_in_channels)
            
        new_conv.weight.copy_(repeated_weight)
        if conv2d.bias is not None:
            new_conv.bias.copy_(conv2d.bias)
    return new_conv

class InflatedResNet(nn.Module):
    def __init__(self, depth=64, scaling_method='linear', **kwargs):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.conv1 = inflate_conv_weights(self.backbone.conv1, new_in_channels=depth, scaling_method=scaling_method)
        self.num_features = 512

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1) # (B, 1, D, H, W) -> (B, D, H, W)
        return self.backbone(x)

class InflatedViT(nn.Module):
    def __init__(self, depth=64, scaling_method='linear', **kwargs):
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.backbone.heads = nn.Identity()
        self.backbone.conv_proj = inflate_conv_weights(self.backbone.conv_proj, new_in_channels=depth, scaling_method=scaling_method)
        self.num_features = 768

    def forward(self, x):
        if x.dim() == 5: x = x.squeeze(1)
        return self.backbone(x)