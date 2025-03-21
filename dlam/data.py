import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from dlam import utils


def read_config(path, stacking_dim=None, selection=None, iselection=None):
    dataset = xr.open_zarr(path)
    features = list(dataset.data_vars)
    feature_names = features

    selection = get_slice(selection)
    iselection = get_slice(iselection)

    dataset = dataset.sel(**selection)
    dataset = dataset.isel(**iselection)

    if stacking_dim:
        dataset = dataset.rename({stacking_dim: "feature_dim"})
        dataset.feature_dim

        feature_names = [feature for feature in features for _ in dataset.feature_dim]

    feature_da = xr.concat(
        [dataset[feature] for feature in features], dim="feature_dim", coords="minimal"
    )
    feature_da.coords["feature_dim"] = ("feature_dim", feature_names)

    return feature_da


class WeatherDataset(Dataset):
    def __init__(self, input, output, selection=None, iselection=None, coords=None):
        self.coords = coords or ["x", "y"]
        self.output = output

        das = [read_config(**inp) for inp in input]
        ds = xr.concat(das, dim="feature_dim", coords="minimal")

        selection = get_slice(selection)
        iselection = get_slice(iselection)

        ds = ds.sel(**selection)
        self.ds = ds.isel(**iselection)

        self.time = ds.time.values

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        batch = utils.AttrDict()

        time = self.time[idx]
        frame = self.ds.sel(time=time)

        stacked_frame = frame.stack(node=self.coords)

        pos = xr.concat(
            [stacked_frame.coords[coord] for coord in self.coords], dim="coord"
        ).T

        batch.pos = torch.Tensor(pos.values)
        batch.time = torch.tensor(int(time) // 1e9, dtype=torch.int64)

        for output_name, output_features in self.output.items():
            if output_features == "all":
                output_features = np.unique(self.ds.feature_dim.values)

            stack = [
                stacked_frame.sel(feature_dim=output_feature)
                for output_feature in output_features
            ]
            batch[output_name] = torch.Tensor(
                xr.concat(stack, dim="feature_dim", coords="minimal").values
            ).T
            __import__("pdb").set_trace()  # TODO delme

        return batch


def dict_to_slice(d):
    return slice(d.get("start", None), d.get("end", None), d.get("step", None))


def get_slice(selection):
    if selection is None:
        return {}

    return {
        k: dict_to_slice(v) if isinstance(v, dict) else v for k, v in selection.items()
    }
