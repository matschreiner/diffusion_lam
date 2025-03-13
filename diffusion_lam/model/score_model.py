import pytorch_lightning as pl
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class NaiveModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = MLP(11, 11, 4)

    def forward(self, batch):
        init_states = self.reshape_and_stack(batch.init_states)
        forcing_states = self.reshape_and_stack(batch.forcing)
        input_ = torch.cat([init_states, forcing_states], dim=-1)

        return self.net(input_)

    def training_step(self, batch):
        out = self.forward(batch)
        loss = torch.nn.functional.mse_loss(out, batch.target_states)
        self.log("loss", loss, prog_bar=True)
        return loss

    def reshape_and_stack(self, tensor):
        tensor = tensor.permute(0, 2, 1, 3)
        shape = tensor.shape
        return tensor.reshape(shape[0], shape[1], -1)
