import pytorch_lightning as pl
import torch

from dlam.model import dpm_solve, noise_schedule


class CosineAlphaSchedule:
    def __init__(self, s=0.008):
        self.s = s
        self.f0 = self.f(torch.tensor(0.0))

    def f(self, t_diff):
        return torch.cos(t_diff * torch.pi * 0.5) ** 2

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

        #  import matplotlib.pyplot as plt
        #
        #  i = torch.linspace(0, 1, 1000)
        #  alpha = self.noise_schedule.get_alpha(i)
        #  signal = alpha**0.5
        #  noise = (1 - alpha) ** 0.5
        #  plt.plot(i, alpha)
        #  plt.plot(i, signal)
        #  plt.plot(i, noise)
        #  plt.plot(i, signal + noise)
        #  plt.show()
        #  plt.plot(i, alpha + (1 - alpha))
        #  __import__("pdb").set_trace()  # TODO delme

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
            t_end=1e-4,
            steps=ode_steps,
            order=1,
            method="singlestep",
            return_intermediate=True,
        )
        return corr, intermediates

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def multiply_first_dim(x, y):
    x_perm = x.movedim(0, -1)
    y_perm = y.movedim(0, -1)
    z_perm = x_perm * y_perm
    return z_perm.movedim(-1, 0)
