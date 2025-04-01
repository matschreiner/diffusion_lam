import pytorch_lightning as pl
import torch

from dlam.model import dpm_solve, noise_schedule


class CosineAlphaSchedule:
    def __init__(self, s=0.008):
        self.s = s
        self.f0 = self.f(torch.tensor(0.0))

    def f(self, t_diff):
        return torch.cos(t_diff * torch.pi * 0.5)

    def get_alpha(self, t_diff):
        return self.f(t_diff) / self.f0


class DDPM(pl.LightningModule):
    def __init__(self, score_model):
        super().__init__()
        self.score_model = score_model
        self.noise_schedule = CosineAlphaSchedule()
        self.loss = []

    def training_step(self, target, _):
        batch_size = target.shape[0]

        epsilon = torch.randn_like(target)
        t_diff = torch.rand([batch_size, 1], device=target.device)
        alpha = self.noise_schedule.get_alpha(t_diff)
        corrupted = multiply_first_dim(alpha**0.5, target) + multiply_first_dim(
            (1 - alpha) ** 0.5, epsilon
        )

        epsilon_hat = self.score_model(corrupted, t_diff)
        print("epsilon_hat", epsilon_hat.std())
        print("epsilon    ", epsilon.std())
        loss = torch.nn.functional.mse_loss(epsilon_hat, epsilon)

        self.log("loss", loss, prog_bar=True, logger=True)
        self.loss.append(loss.item())
        return loss

    def sample(self, batch, ode_steps):
        noise_schedule = dpm_solve.NoiseScheduleVP(schedule="cosine")

        def forward(corr, t_diff):
            t_diff.unsqueeze_(-1)
            epsilon_hat = self.score_model(corr, t_diff)

            return epsilon_hat

        wrapped_model = dpm_solve.model_wrapper(forward, noise_schedule)
        dpm_solver = dpm_solve.DPM_Solver(
            wrapped_model,
            noise_schedule,
            algorithm_type="dpmsolver++",
        )

        corr = torch.randn_like(batch, device=batch.device)
        corr, intermediates = dpm_solver.sample(
            corr,
            t_end=1e-3,
            steps=ode_steps,
            order=1,
            method="singlestep",
            return_intermediate=True,
        )
        return corr, intermediates

    #  def sample(self, batch, sde_steps=1000):
    #      x = torch.randn_like(batch)
    #      alpha_bars = [
    #          self.noise_schedule.get_alpha(torch.tensor(i / sde_steps, device=x.device))
    #          for i in range(sde_steps + 1)
    #      ]
    #      alpha_bars[0] = torch.tensor(1.0, device=x.device)
    #      alphas = [alpha_bars[i] / alpha_bars[i + 1] for i in range(sde_steps)]
    #      xs = [x]
    #      for t in reversed(range(1, sde_steps + 1)):
    #
    #          t_diff = torch.full((len(x), 1), t / sde_steps)
    #
    #          epsilon_hat = self.score_model(x, t_diff)
    #          alpha_bar = self.noise_schedule.get_alpha(t_diff)
    #          alpha_bar_m1 = self.noise_schedule.get_alpha(t_diff - 1 / sde_steps)
    #          alpha = alpha_bar / alpha_bar_m1
    #
    #          x = (
    #              1
    #              / (alpha**0.5)
    #              * (x - (1 - alpha) / ((1 - alpha_bar) ** 0.5) * epsilon_hat)
    #          )
    #
    #          if t > 1:
    #              variance = 0.0001
    #              std = variance ** (0.5)
    #              x += std * torch.randn_like(x)
    #
    #          x = x.detach()
    #          xs.append(x)
    #
    #      return x, xs


def multiply_first_dim(x, y):
    x_perm = x.movedim(0, -1)
    y_perm = y.movedim(0, -1)
    z_perm = x_perm * y_perm
    return z_perm.movedim(-1, 0)
