import matplotlib.pyplot as plt
import torch
import xarray as xr
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, path, selection=None, iselection=None, variables=None):

        selection = get_slice(selection)
        iselection = get_slice(iselection)

        data = xr.open_zarr(path)
        data = data.sel(**selection)
        data = data.isel(**iselection)

        self.data = data
        self.variables = variables or list(data.data_vars)

    def __len__(self):
        return len(self.data.time)

    def __getitem__(self, idx):
        frame = self.data.isel(time=idx)
        frame.stack(features=self.variables)
        frame.stacked(features=self.variables)
        __import__("pdb").set_trace()  # TODO delme kj:w


def dict_to_slice(d):
    return slice(d.get("start", None), d.get("end", None), d.get("step", None))


def get_slice(selection):
    if selection is None:
        return {}

    return {
        k: dict_to_slice(v) if isinstance(v, dict) else v for k, v in selection.items()
    }
