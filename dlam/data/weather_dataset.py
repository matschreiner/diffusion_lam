import os
from functools import lru_cache

import matplotlib.pyplot as plt
import neural_lam
import numpy as np
import torch
from neural_lam.datastore.mdp import MDPDatastore

import dlam.utils
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


class WeatherDatasetFromDatastore(neural_lam.weather_dataset.WeatherDataset):
    def __init__(self, datastore, precision=torch.float32):
        datastore = MDPDatastore(datastore) if isinstance(datastore, str) else datastore

        super().__init__(datastore)
        self.xy = torch.tensor(datastore.get_xy("state", stacked=False))
        self.static = torch.tensor(
            datastore.get_dataarray(
                category="static", split=None, standardize=True
            ).values
        )
        self.precision = precision
        self.boundary_mask = torch.tensor(datastore.boundary_mask.values).to(precision)
        self.interior_mask = 1 - self.boundary_mask

    def add_graph(self, create_graph):
        self.graph = create_graph(self.xy)

    @lru_cache(maxsize=20)
    def __getitem__(self, index):
        cond_states, target_states, forcing, times = super().__getitem__(index)

        item = {}
        item["static"] = self.static.to(self.precision)
        item["xy"] = self.xy.to(self.precision)
        item["forcing"] = forcing.to(self.precision)
        item["cond"] = cond_states.to(self.precision)
        item["target"] = target_states.to(self.precision)[0]
        item["time"] = times.to(self.precision)
        item["bounday_mask"] = self.boundary_mask
        item["interior_mask"] = self.interior_mask

        if hasattr(self, "graph"):
            item["graph"] = self.graph

        return AttrDict(item)


class WeatherDatasetZarr(torch.utils.data.Dataset):
    def __init__(self, storage, precision=torch.float32):
        self.precision = precision
        data = xarray.open_zarr(os.path.join(storage, "height_levels.zarr")).isel(
            x=slice(0, 10), y=slice(0, 10)
        )
        static = xarray.open_zarr(os.path.join(storage, "static.zarr")).isel(
            x=slice(0, 10), y=slice(0, 10)
        )
        #  data = xarray.open_zarr(os.path.join(storage, "height_levels.zarr"))
        #  static = xarray.open_zarr(os.path.join(storage, "static.zarr"))

        self.xy = np.stack(
            np.meshgrid(data.x.values, data.y.values, indexing="ij"), axis=-1
        )

        del data["danra_projection"]
        data = data.to_stacked_array("feature", ["time", "x", "y"])
        self.data = data.stack(dims=["x", "y"])

        self.std = torch.tensor(self.data.values.std(axis=(0, 2))).to(self.precision)
        self.mean = torch.tensor(self.data.values.mean(axis=(0, 2))).to(self.precision)

        static = static.to_stacked_array("feature", ["x", "y"])
        self.static = torch.tensor(static.stack(dims=["x", "y"]).values).T.to(
            self.precision
        )

        self.static[:, 1] = (
            self.static[:, 1] - self.static[:, 1].mean()
        ) / self.static[:, 1].std()

    def __len__(self):
        return 1
        #  return len(self.data) - 2

    def __getitem__(self, idx):
        item = {}
        item["static"] = self.static
        item["xy"] = self.xy

        cond1 = torch.tensor(self.data.isel(time=idx).values).T.to(self.precision)
        cond2 = torch.tensor(self.data.isel(time=idx + 1).values).T.to(self.precision)
        target = torch.tensor(self.data.isel(time=idx + 2).values).T.to(self.precision)

        cond1 = (cond1 - self.mean) / self.std
        cond2 = (cond2 - self.mean) / self.std
        target = (target - self.mean) / self.std

        cond = torch.stack([cond1, cond2], dim=0)

        item["cond"] = cond
        item["target"] = target
        if hasattr(self, "graph"):
            item["graph"] = self.graph

        return dlam.utils.AttrDict(item)

    def add_graph(self, create_graph):
        self.graph = create_graph(self.xy)


if __name__ == "__main__":
    ds = WeatherDatasetZarr("storage")
    d = ds[0]
    #  dataset = WeatherDataset("storage/mini.zarr/")
    #  datalist = dataset[:]
    #
    #  def fn(ax, data):
    #      ax.imshow(data.numpy(), cmap="viridis")
    #      ax.set_title("t2m")
    #
    #  ani = animate(
    #      datalist,
    #      fn,
    #  )
    #  plt.show()
