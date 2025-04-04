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

    def forward(self, batch, t_diff):
        x = torch.cat([batch.corr, t_diff], dim=1)

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

    def forward(self, batch, t_diff):
        x = batch.corr
        cond = batch.cond.unsqueeze(1)
        x = torch.cat([x, cond, t_diff], dim=1)

        self.net(x)
        x = self.net(x)
        return x
