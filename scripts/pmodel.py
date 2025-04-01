import pytorch_lightning as pl
import torch


class Schedule:
    def __init__(self, diffusion_steps=1000):
        s = 0.008
        timesteps = torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
        schedule = (
            torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2) ** 2
        )

        self.baralphas = schedule / schedule[0]
        self.betas = 1 - self.baralphas / torch.concatenate(
            [self.baralphas[0:1], self.baralphas[0:-1]]
        )
        self.alphas = 1 - self.betas

    def get_alphabar(self, idx):
        return self.baralphas[idx]

    def get_beta(self, idx):
        return self.betas[idx]

    def get_alpha(self, idx):
        return self.alphas[idx]


class DDPM(pl.LightningModule):
    def __init__(self, score_model, diffusion_steps=1000):
        super().__init__()
        self.score_model = score_model
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = Schedule(diffusion_steps=self.diffusion_steps)

    def get_noise(self, x, t):
        eps = torch.randn_like(x)
        noised = (self.noise_schedule.get_alphabar(t) ** 0.5).repeat(
            1, x.shape[1]
        ) * x + ((1 - self.noise_schedule.get_alphabar(t)) ** 0.5).repeat(
            1, x.shape[1]
        ) * eps
        return noised, eps

    def training_step(self, x):
        timesteps = torch.randint(0, self.diffusion_steps, size=[len(x), 1])
        noised, eps = self.get_noise(x, timesteps)
        predicted_noise = self.score_model(noised, timesteps)

        loss = torch.nn.functional.mse_loss(predicted_noise, eps)
        self.log("loss", loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=500
        )
        return [optimizer], [scheduler]


class DiffusionBlock(torch.nn.Module):
    def __init__(self, nunits):
        super(DiffusionBlock, self).__init__()
        self.linear = torch.nn.Linear(nunits, nunits)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        return x


class ScoreModel(torch.nn.Module):
    def __init__(self, nfeatures=2, nblocks=2, nunits=64):
        super().__init__()

        self.inblock = torch.nn.Linear(nfeatures + 1, nunits)
        self.midblocks = torch.nn.ModuleList(
            [DiffusionBlock(nunits) for _ in range(nblocks)]
        )
        self.outblock = torch.nn.Linear(nunits, nfeatures)

    def forward(self, x, t):
        val = torch.hstack([x, t])
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val)
        return val
