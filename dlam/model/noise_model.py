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


class NoiseModelCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, batch, t_diff):
        x = batch.corr
        t_diff = t_diff.repeat(1, 1, x.shape[-2], x.shape[-1])

        x = torch.concat([x, t_diff], dim=1).to(torch.float32)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class ConditionalNoiseModelCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        in_channels = n_features * 2 + 1

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, n_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, batch, t_diff):
        x = batch.corr
        cond = batch.cond

        t_diff = t_diff.repeat(1, 1, x.shape[-2], x.shape[-1])

        x = torch.concat([x, cond, t_diff], dim=1).to(torch.float32)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)

        return x
