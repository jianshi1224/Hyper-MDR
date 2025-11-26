import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperedgeGenerator(nn.Module):
    """
    Learnable hyperedge gating network
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, nodes):
        """
        nodes: (N, D)
        returns hyperedge adjacency A: (N, N)
        """
        N, D = nodes.shape
        n1 = nodes.unsqueeze(1).expand(N, N, D)
        n2 = nodes.unsqueeze(0).expand(N, N, D)
        pair = torch.cat([n1, n2], dim=-1)
        A = torch.sigmoid(self.fc(pair)).squeeze(-1)
        return A
