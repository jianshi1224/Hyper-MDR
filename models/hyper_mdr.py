import torch
import torch.nn as nn
from .nodes import NodeEmbedding
from .hyperedge import HyperedgeGenerator
from .hypergnn import HyperGNNLayer
from .controller import MetaPolicyController

class HyperMDR(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        self.node_proj = NodeEmbedding(dim, dim)
        self.hyperedge_gen = HyperedgeGenerator(dim)
        self.layers = nn.ModuleList([HyperGNNLayer(dim) for _ in range(5)])
        self.controller = MetaPolicyController()

        self.cls_head = nn.Linear(dim, dim)

    def forward(self, visual, text, prototype, state, hx):
        nodes = torch.cat([
            self.node_proj(visual),
            self.node_proj(text),
            self.node_proj(prototype)
        ], dim=0)

        wv, wl, K, hx = self.controller(state, hx)
        fused_nodes = (wv * nodes + wl * nodes)

        A = self.hyperedge_gen(fused_nodes)

        h = fused_nodes
        for i in range(K.item()):
            h = self.layers[i](h, A)

        logits = self.cls_head(h)

        return logits, h, A, hx
