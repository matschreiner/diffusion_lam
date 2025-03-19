import torch
import xarray as xr
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, inputs, selection=None, iselection=None):
        data = read_inputs(inputs)

        selection = get_slice(selection)
        iselection = get_slice(iselection)

        data = data.sel(**selection)
        data = data.isel(**iselection)

        self.data = data

    def __len__(self):
        return len(self.data.time)

    def __getitem__(self, idx):
        frame = self.data.isel(time=idx)
        return torch.Tensor(state_feature.values)


def read_inputs(inputs):
    [ds1, ds2] = [read_input(**input_) for input_ in inputs]
    __import__("pdb").set_trace()  # TODO delme
    xr.concat(dss, dim="feature")
    return xr.concat(dss, dim="feature")


def read_input(path, features=None, stacking_dims=None):
    ds = xr.open_zarr(path)
    #  ds = xr.broadcast(ds)[0]

    stacking_dims = stacking_dims or []
    sample_dims = [dim for dim in ds.dims.keys() if dim not in stacking_dims]
    stacked_array = ds[features].to_stacked_array("feature", sample_dims=sample_dims)
    stacked_ds = xr.Dataset({"feature": stacked_array})
    __import__("pdb").set_trace()  # TODO delme

    return stacked_array


def dict_to_slice(d):
    return slice(d.get("start", None), d.get("end", None), d.get("step", None))


def get_slice(selection):
    if selection is None:
        return {}

    return {
        k: dict_to_slice(v) if isinstance(v, dict) else v for k, v in selection.items()
    }
