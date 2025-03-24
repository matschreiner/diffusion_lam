import pytorch_lightning as pl
import torch

from dlam.model import noise_schedule


class DDPM(pl.LightningModule):
    def __init__(self, score_model, power=2.0):
        super().__init__()
        self.score_model = score_model
        self.noise_schedule = noise_schedule.PolynomialSchedule(power=power)
        self.save_hyperparameters()

    def training_step(self, batch, _):

        corrupted, epsilon, t_diff = self.get_corrupted(batch.target.state)

        epsilon_hat = self.score_model(corrupted, t_diff)
        loss = torch.nn.functional.mse_loss(epsilon, epsilon_hat)

        self.log("loss", loss, prog_bar=True, logger=True)

        return loss.mean()

    def get_corrupted(self, x):
        batch_size = x.shape[0]

        t_diff = torch.rand(batch_size, device=x.device)
        alpha = self.noise_schedule.get_alpha(t_diff)

        x = x.permute(1, 2, 3, 0)
        epsilon = torch.randn_like(x, device=x.device)
        corrupted = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * epsilon

        return corrupted.permute(3, 0, 1, 2), epsilon.permute(3, 0, 1, 2), t_diff

    def on_before_zero_grad(self, *_, **__):  # pylint:disable=unused-argument
        self.ema.update(self.score_model.parameters())

    #  def sample(self, batch, ode_steps):
    #      n_disc = 1000
    #
    #      alphas_cumprod = self.noise_schedule.get_alpha(torch.linspace(0, 1, n_disc))
    #      sample_tracker = SampleTracker()
    #      noise_schedule = dpm_solve.NoiseScheduleVP(
    #          "discrete",
    #          alphas_cumprod=alphas_cumprod,
    #      )
    #
    #      def forward(x, t_diff):
    #          batch["corr"].x = x
    #          batch["t_diff"] = t_diff / n_disc - 1 / ode_steps
    #
    #          sample_tracker.add_frame(x)
    #          epsilon_hat = self.score_model(batch)
    #
    #          return epsilon_hat.x
    #
    #      wrapped_model = dpm_solve.model_wrapper(forward, noise_schedule)
    #      dpm_solver = dpm_solve.DPM_Solver(wrapped_model, noise_schedule)
    #
    #      batch["corr"] = get_epsilon_like(batch["cond"])
    #      batch["corr"].x = dpm_solver.sample(batch["corr"].x, ode_steps)
    #
    #      return batch["corr"], sample_tracker.get_traj()
