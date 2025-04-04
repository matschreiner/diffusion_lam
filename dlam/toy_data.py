import sklearn
import torch

from dlam.utils import AttrDict


class HalfmoonDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, noise=0.1, seed=42, normalize=False):
        self.n_samples = n_samples
        self.noise = noise
        self.seed = seed

        data, cls = sklearn.datasets.make_moons(
            n_samples=n_samples, noise=noise, random_state=seed
        )
        if normalize:
            data = (data - data.mean(axis=0)) / data.std(axis=0)

        self.data = torch.tensor(data, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
        #  return AttrDict({"target": self.data[idx], "cond": self.cls[idx]})
