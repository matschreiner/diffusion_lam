import torch
from torch import nn

from dlam.model import mlp


class NoiseModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layers):
        super().__init__()

        self.net = mlp.MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
        )

    def forward(self, x, t_diff):
        x = torch.cat([x, t_diff], dim=1)

        self.net(x)
        x = self.net(x)
        return x


class ConditionalNoiseModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layers):
        super().__init__()

        self.net = mlp.MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
        )

    def forward(self, x, t_diff):
        x = torch.cat([x, t_diff], dim=1)

        self.net(x)
        x = self.net(x)
        return x
