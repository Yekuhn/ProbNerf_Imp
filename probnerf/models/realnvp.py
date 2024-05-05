import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, reverse=False):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim // 2  # Assuming input_dim is even and we split it into two equal halves
        self.reverse = reverse

        # MLP for generating shift and scale parameters
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim * 2)  # Outputs for both shift (t) and scale (s)
        )

    def forward(self, x):
        if not self.reverse:
            x1, x2 = x.chunk(2, dim=1)
        else:
            x2, x1 = x.chunk(2, dim=1)

        st = self.net(x1)
        shift, log_scale = st.chunk(2, dim=1)
        scale = torch.exp(log_scale)  # Ensuring scale is positive
        x2 = x2 * scale + shift

        if not self.reverse:
            return torch.cat([x1, x2], dim=1)
        else:
            return torch.cat([x2, x1], dim=1)

def random_permute(x):
    idx = torch.randperm(x.nelement())
    return x.view(-1)[idx].view(x.size())

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_blocks=4):
        super(RealNVP, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Alternate the reversal of the split for each block
            reverse = (i % 2 == 1)
            self.blocks.append(AffineCouplingLayer(input_dim, hidden_dim, reverse))

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Apply random permutation after each pair of blocks
            if (i + 1) % 2 == 0:  # After every two blocks
                x = random_permute(x)
        return x
