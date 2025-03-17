import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden):
        super().__init__()

        modules = [torch.nn.Linear(input_dim, hidden_dim)]

        for i in range(num_hidden):
            modules.append(torch.nn.SiLU())
            layer_dim = hidden_dim if i < num_hidden - 1 else output_dim
            modules.append(torch.nn.Linear(hidden_dim, layer_dim))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
