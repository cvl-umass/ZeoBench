import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, n_feature=5, n_hidden=200):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.num_features = n_hidden

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return x

def make_mlp(n_feature=5, n_hidden=200, **kwargs):
    return MLP(n_feature, n_hidden)
