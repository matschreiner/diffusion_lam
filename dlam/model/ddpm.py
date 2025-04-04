import pytorch_lightning as pl
import torch

from dlam.model import dpm_solve


class CosineAlphaSchedule:
    def __init__(self):
        self.f0 = self.f(torch.tensor(0.0))

    def f(self, t):
        s = 0.008
        t = torch.tensor(t)
        return torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2

    def get_alpha(self, t):
        return self.f(t) / self.f0


class DDPM(pl.LightningModule):
    def __init__(self, score_model, noise_schedule):
        super().__init__()
        self.score_model = score_model
        self.noise_schedule = noise_schedule
        #  self.noise_schedule = CosineAlphaSchedule()
        self.loss_fn = torch.nn.MSELoss()
        self.loss = []

    def training_step(self, x):
        t_shape = [x.shape[0]] + [1] * (x.dim() - 1)
        t = torch.rand(size=t_shape)

        corr, eps = self.get_corrupted(x, t)
        predicted_noise = self.score_model(corr, t)

        loss = self.loss_fn(predicted_noise, eps)
        self.log("loss", loss, prog_bar=True, logger=True)

        return loss

    def get_corrupted(self, x, t):
        eps = torch.randn(size=x.shape)
        alpha = self.noise_schedule.get_alpha(t)
        corr = alpha**0.5 * x + (1 - alpha) ** 0.5 * eps
        return corr, eps

    def sde_sample(self, example, steps=100):
        with torch.no_grad():
            x = torch.randn_like(example)
            xt = [x]
            for t in range(steps - 1, 0, -1):
                t = t / steps
                tm1 = t - 1 / steps
                print("model", t, tm1)

                t_diff = torch.full([len(x), 1], t)
                predicted_noise = self.score_model(x, t_diff)
                alphabar = self.noise_schedule.get_alpha(t)
                alphabarm1 = self.noise_schedule.get_alpha(tm1)
                alpha = alphabar / alphabarm1
                beta = 1 - alpha

                x = (
                    1
                    / (alpha**0.5)
                    * (x - (1 - alpha) / ((1 - alphabar) ** 0.5) * predicted_noise)
                )
                if t > 1 / steps:
                    variance = beta
                    std = variance ** (0.5)
                    x += std * torch.randn_like(example)
                xt += [x]

            return x, xt
