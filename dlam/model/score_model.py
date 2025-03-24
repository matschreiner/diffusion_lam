import pytorch_lightning as pl
import torch

from dlam.model import mlp


class NaiveModel(pl.LightningModule):
    def __init__(self, data_dim):
        super().__init__()
        flat_dim = data_dim[0] * data_dim[1]
        self.net = mlp.MLP(flat_dim, flat_dim, 200)

    def forward(self, batch):
        cond = batch
        shape = cond.state.shape
        input_ = cond.state.reshape(1, -1)

        output = input_.reshape(shape)

        output = self.net(input_)
        output = output.reshape(shape)
        self.last = output

        return output

    def training_step(self, batch, _):
        target = batch.target
        cond = batch.cond
        output = self.forward(cond)

        loss = torch.nn.functional.mse_loss(output, target.state)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
