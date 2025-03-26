import sklearn
import torch


class HalfmoonDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, noise=0.1, seed=42):
        self.n_samples = n_samples
        self.noise = noise
        self.seed = seed

        data, _ = sklearn.datasets.make_moons(
            n_samples=n_samples, noise=noise, random_state=seed
        )

        data = torch.tensor(data, dtype=torch.float32)
        self.data = (data - data.mean()) / data.std()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
