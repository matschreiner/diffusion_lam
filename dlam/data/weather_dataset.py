import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray

from dlam.utils import AttrDict
from dlam.vis.animate import animate


class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path, len=None):
        data = xarray.open_zarr(zarr_path)
        #  t2m = self.data.t2m.to_numpy()

        vs = [
            "r",
            "t",
            "u",
            "v",
        ]
        data = [v.as_numpy() for v in data.values()]
        data = np.concatenate(data, axis=1)
        data = torch.tensor(data, dtype=torch.float32)

        #  features = self.data.to_array("feature").transpose("time", "feature", "y", "x")
        #  nps = [torch.tensor(self.data[v].values) for v in vs]
        #  data = torch.stack(nps, dim=1)
        std = torch.std(data, dim=(0, 2, 3), keepdim=True)
        mean = torch.mean(data, dim=(0, 2, 3), keepdim=True)
        self.data = (data - mean) / std
        self.len = len

    def __len__(self):
        if self.len:
            return self.len
        return len(self.data) - 1

    def __getitem__(self, idx):
        cond = self.data[idx]
        target = self.data[idx + 1]

        return AttrDict({"target": target, "cond": cond})


if __name__ == "__main__":
    dataset = WeatherDataset("storage/mini.zarr/")
    datalist = dataset[:]

    def fn(ax, data):
        ax.imshow(data.numpy(), cmap="viridis")
        ax.set_title("t2m")

    ani = animate(
        datalist,
        fn,
    )
    plt.show()
