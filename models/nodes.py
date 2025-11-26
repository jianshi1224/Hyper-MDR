import torch
import torch.nn as nn

class NodeEmbedding(nn.Module):
    """
    Embedding for visual / text / prototype nodes
    """
    def __init__(self, in_dim=512, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))
