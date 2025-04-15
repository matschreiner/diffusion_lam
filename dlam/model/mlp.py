import torch


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, hidden_layers=1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        modules = [torch.nn.Linear(in_dim, hidden_dim, bias=True)]

        for i in range(hidden_layers):
            modules.append(torch.nn.SiLU())
            layer_in_dim = hidden_dim
            layer_out_dim = hidden_dim if i < hidden_layers - 1 else out_dim
            modules.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=True))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
