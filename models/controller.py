import torch
import torch.nn as nn

class MetaPolicyController(nn.Module):
    """
    LSTM controller that outputs:
    - visual weight α_v
    - text weight α_l
    - #hyperGNN layers K
    """
    def __init__(self, state_dim=128, hidden=128):
        super().__init__()

        self.lell = nn.LSTMCell(state_dim, hidden)

        self.fc_wv = nn.Linear(hidden, 1)
        self.fc_wl = nn.Linear(hidden, 1)
        self.fc_layer = nn.Linear(hidden, 1)

    def forward(self, state, hx):
        hx = self.lell(state, hx)
        h = hx[0]

        wv = torch.sigmoid(self.fc_wv(h))
        wl = torch.sigmoid(self.fc_wl(h))
        K = torch.clamp(torch.round(3 + self.fc_layer(h)), 1, 5).int()

        return wv, wl, K, hx
