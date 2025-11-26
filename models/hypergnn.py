import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperGNNLayer(nn.Module):
    """
    Node→Hyperedge→Node message passing with gated update
    """
    def __init__(self, dim):
        super().__init__()

        self.node2edge = nn.Linear(dim, dim)
        self.edge2node = nn.Linear(dim, dim)

        self.update_gate = nn.Linear(dim * 2, dim)
        self.reset_gate = nn.Linear(dim * 2, dim)
        self.h_candidate = nn.Linear(dim * 2, dim)

    def forward(self, nodes, A):
        N, D = nodes.size()

        edge_msg = torch.matmul(A, self.node2edge(nodes))
        node_msg = torch.matmul(A.t(), self.edge2node(edge_msg))

        concat = torch.cat([nodes, node_msg], dim=-1)
        z = torch.sigmoid(self.update_gate(concat))
        r = torch.sigmoid(self.reset_gate(concat))
        h_hat = torch.tanh(self.h_candidate(torch.cat([nodes * r, node_msg], dim=-1)))

        return (1 - z) * nodes + z * h_hat
