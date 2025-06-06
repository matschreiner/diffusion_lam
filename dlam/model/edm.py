import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from dlam import utils
from dlam.model import ema

# Code lifted from https://github.com/NVlabs/edm


class EDM(pl.LightningModule):
    def __init__(
        self,
        noise_model,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=1,
    ):
        super().__init__()

        if isinstance(noise_model, dict):
            self.noise_model = utils.get_component(noise_model)
        else:
            self.noise_model = noise_model

        if hasattr(self.noise_model, "create_graph"):
            self.create_graph = self.noise_model.create_graph

        self.precond_model = EDMPrecond(noise_model)

        # loss params
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.save_hyperparameters()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["noise_model"] = self.noise_model

    def training_step(self, batch):
        loss = self.get_loss(batch).mean()
        self.log("loss", loss.mean(), prog_bar=True, logger=True)
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
            logger=True,
        )
        self.log("log_loss", loss.mean().log10(), logger=True)

        return loss.mean()

    def get_loss(self, batch):
        target = batch.target

        rnd_normal = get_random(target, torch.randn)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(target) * sigma

        batch.corr = target + n
        D_yn = self.precond_model(batch, sigma)
        loss = weight * ((D_yn - target) ** 2)
        return loss

    def sample(self, batch, steps=18, **kwargs):
        sample, intermediate = edm_sampler(
            self.precond_model,
            batch,
            num_steps=steps,
            **kwargs,
        )
        return sample, intermediate

    def create_graph(self):
        self.noise_model.create_graph()


def get_random(target, rand_fn):
    shape = [target.shape[0]] + [1] * (target.dim() - 1)
    return rand_fn(size=shape, device=target.device)


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        noise_model,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=1,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.noise_model = noise_model

    def forward(self, batch, sigma):
        skip_corr = batch.corr.to(torch.float32).clone()

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        batch.corr = c_in * skip_corr

        F_x = self.noise_model(
            batch,
            c_noise.view(-1),
        )

        D_x = c_skip * skip_corr + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def edm_sampler(
    precond_model,
    batch,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    sigma_min = max(sigma_min, precond_model.sigma_min)
    sigma_max = min(sigma_max, precond_model.sigma_max)
    latents = torch.randn_like(batch.target)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [precond_model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]
    intermediate = [x_next]
    with torch.no_grad():
        for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))):
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )

            t_hat = precond_model.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            t_hat = get_random(x_hat, torch.ones) * t_hat
            t_next = get_random(x_hat, torch.ones) * t_next

            batch.corr = x_hat

            denoised = precond_model(batch, t_hat).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            #  Apply 2nd order correction.
            if i < num_steps - 1:
                batch.corr = x_next
                denoised = precond_model(batch, t_next).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            intermediate.append(x_next)

    return x_next, intermediate
