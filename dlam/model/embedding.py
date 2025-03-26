import numpy as np
import torch


class PositionalEncoder(torch.nn.Module):
    def __init__(self, dim, max_length=10):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.max_length = max_length
        self.max_rank = dim // 2

    def forward(self, x):
        encodings = [
            self.positional_encoding(x, rank) for rank in range(1, self.max_rank + 1)
        ]
        encodings = torch.cat(
            encodings,
            axis=1,
        )
        return encodings

    def positional_encoding(self, x, rank):
        x = torch.clamp(x, 0, self.max_length)
        sin = torch.sin(x / self.max_length * rank * np.pi)
        cos = torch.cos(x / self.max_length * rank * np.pi)
        return torch.stack((cos, sin), axis=1)
