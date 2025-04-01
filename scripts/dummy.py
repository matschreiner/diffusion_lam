import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MinimalDiffusionModel(pl.LightningModule):
    def __init__(self, data_shape, timesteps=1000, lr=1e-3):
        super().__init__()
        self.data_shape = data_shape
        self.timesteps = timesteps
        self.lr = lr
        flat_dim = 1
        for d in data_shape:
            flat_dim *= d
        self.model = SimpleMLP(flat_dim + 1, 128, flat_dim)
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def forward(self, x, t):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        t = t.unsqueeze(-1)
        model_input = torch.cat([x_flat, t], dim=1)
        noise_pred = self.model(model_input)
        return noise_pred

    def training_step(self, batch, batch_idx):
        (x,) = batch
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        noise = torch.randn_like(x)
        alphas_t = self.alphas_cumprod[t].view(batch_size, *([1] * (len(x.shape) - 1)))
        sqrt_alphas = torch.sqrt(alphas_t)
        sqrt_one_minus_alphas = torch.sqrt(1 - alphas_t)
        x_noisy = sqrt_alphas * x + sqrt_one_minus_alphas * noise
        t_norm = t.float() / self.timesteps
        noise_pred = self(x_noisy, t_norm)
        x_flat = x.view(batch_size, -1)
        noise = noise.view(batch_size, -1)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sample(self, n_samples):
        x = torch.randn(n_samples, *self.data_shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((n_samples,), t, device=x.device, dtype=torch.long)
            t_norm = t_tensor.float() / self.timesteps
            predicted_noise = self(x, t_norm)
            predicted_noise = predicted_noise.view(n_samples, *self.data_shape)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            coef = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = 1 / torch.sqrt(alpha_t) * (x - coef * predicted_noise)
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        return x


def get_dummy_dataloader(data_shape, batch_size=32, num_samples=1000):
    data = torch.randn(num_samples, *data_shape)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size)


if __name__ == "__main__":
    data_shape = (28, 28)
    model = MinimalDiffusionModel(data_shape)
    dataloader = get_dummy_dataloader(data_shape)
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, dataloader)
    samples = model.sample(16)
