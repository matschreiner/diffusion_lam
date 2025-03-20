import torch
import xarray as xr
from torch.utils.data import Dataset

from dlam import utils


class WeatherDataset(Dataset):
    def __init__(self, data, selection=None, iselection=None, coords=None):
        self.coords = coords or ["x", "y"]

        self.ds_dict = {}
        for ds_name, ds_config in data.items():

            ds = read_configs(ds_config)

            selection = get_slice(selection)
            iselection = get_slice(iselection)

            selection = {k: v for k, v in selection.items() if k in ds.dims}
            iselection = {k: v for k, v in iselection.items() if k in ds.dims}

            ds = ds.sel(**selection)
            ds = ds.isel(**iselection)

            if "time" in ds:
                self.time = ds.time

            self.ds_dict[ds_name] = ds

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        time = self.time[idx]

        batch = utils.AttrDict()
        batch.time = torch.tensor(time.values.item() / 1e9)

        for feature, ds in self.ds_dict.items():
            if "time" in ds.dims:
                ds = ds.sel(time=time)

            stacked = ds.stack(node=("x", "y")).features.transpose(
                "node", "feature_dim"
            )

            batch[feature] = torch.tensor(stacked.values)

        pos = xr.concat(
            [stacked.coords[coord] for coord in self.coords], dim="coord"
        ).transpose("node", "coord")
        batch["pos"] = torch.tensor(pos.values)

        return batch


def read_configs(inputs):
    das = [read_config(**input_) for input_ in inputs]
    dataarray = xr.concat(das, dim="feature_dim", coords="minimal")
    dataset = xr.Dataset({"features": dataarray})

    return dataset


def read_config(path, features=None, stacking_dim=None):
    dataset = xr.open_zarr(path)
    features = features or list(dataset.data_vars)

    if stacking_dim:
        dataset = dataset.rename({stacking_dim: "feature_dim"})
        # coords of feature dim are meaningless
        dataset = dataset.drop("feature_dim")

    feature_da = xr.concat(
        [dataset[feature] for feature in features], dim="feature_dim", coords="minimal"
    )

    return feature_da


def dict_to_slice(d):
    return slice(d.get("start", None), d.get("end", None), d.get("step", None))


def get_slice(selection):
    if selection is None:
        return {}

    return {
        k: dict_to_slice(v) if isinstance(v, dict) else v for k, v in selection.items()
    }
