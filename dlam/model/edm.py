import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

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
        self.noise_model = noise_model
        self.model = EDMPrecond(noise_model)

        # loss params
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def training_step(self, batch):
        loss = self.get_loss(batch).mean()
        self.log("loss", loss.mean(), prog_bar=True, logger=True)
        return loss.mean()

    def get_loss(self, batch):
        target = batch.target

        rnd_normal = torch.randn([target.shape[0], 1], device=target.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(target) * sigma

        batch.corr = target + n
        D_yn = self.model(batch, sigma)
        loss = weight * ((D_yn - target) ** 2)
        return loss

    def sample(self, batch, steps=18, **kwargs):
        sample, intermediate = edm_sampler(
            self.model,
            batch,
            num_steps=steps,
            **kwargs,
        )
        return sample, intermediate


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        model,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, batch, sigma):
        skip_corr = batch.corr.to(torch.float32).clone()

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        batch.corr = c_in * skip_corr
        F_x = self.model(
            batch,
            c_noise,
        )

        D_x = c_skip * skip_corr + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def edm_sampler(
    net,
    batch,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    latents = torch.randn_like(batch.target)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
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
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            t_hat = torch.ones((len(x_hat), 1), device=x_hat.device) * t_hat
            t_next = torch.ones((len(x_hat), 1), device=x_hat.device) * t_next

            batch.corr = x_hat

            denoised = net(batch, t_hat).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            #  Apply 2nd order correction.
            if i < num_steps - 1:
                batch.corr = x_next
                denoised = net(batch, t_next).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            intermediate.append(x_next)

    return x_next, intermediate
