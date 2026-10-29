import torch.nn as nn
from monai.networks.nets import resnet18, resnet50

def resnet18_3d(method=None, *args, **kwargs):
    """
    Factory function for MONAI's 3D ResNet18.
    """
    if "kwargs" in kwargs:
        inner_kwargs = kwargs.pop("kwargs")
        kwargs.update(inner_kwargs)
        
    spatial_dims = kwargs.pop("spatial_dims", 3)
    n_input_channels = kwargs.pop("in_channels", 1)
    
    model = resnet18(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=1000,
        **kwargs 
    )
    
    model.num_features = model.fc.in_features
    return model

def resnet50_3d(method=None, *args, **kwargs):
    if "kwargs" in kwargs:
        inner_kwargs = kwargs.pop("kwargs")
        kwargs.update(inner_kwargs)

    spatial_dims = kwargs.pop("spatial_dims", 3)
    n_input_channels = kwargs.pop("in_channels", 1)
    
    model = resnet50(
        spatial_dims=spatial_dims, 
        n_input_channels=n_input_channels, 
        num_classes=1000,
        **kwargs
    )
    model.num_features = model.fc.in_features
    return model