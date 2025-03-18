import matplotlib.pyplot as plt
import torch
import xarray as xr
from torch.utils.data import Dataset

#
#  from torch
#
#  import yaml
#
#  datastore_path = "/home/masc/mllam/diffusion-lam/experiments/test/danra.datastore.yaml"
#  yaml_datastore = yaml.safe_load(open(datastore_path, "r"))
#  pprint(yaml_datastore)


class WeatherDataset(Dataset):
    def __init__(self, path, selection=None, iselection=None):

        selection = selection or {}
        iselection = iselection or {}

        data = xr.open_zarr(path)
        data = data.sel(**selection)
        data = data.isel(**iselection)
        self.data = data

        __import__("pdb").set_trace()  # TODO delme kj:w

    def stack_all(self, frame):
        variables = ["u", "v"]
        stacked_vars = []

        for var in variables:
            data = frame[var]
            feature_dims = data.dims[:-2]
            stacked = data.stack(feature=feature_dims)

            stacked_vars.append(stacked)
        return xr.concat(stacked_vars, dim="feature")

    def __len__(self):
        return len(self.data.time)

    def __getitem__(self, idx):
        frame = self.data.isel(time=idx, **self.slicing)
        frame = self.stack_all(frame)

        return frame


#  ds = Dataset()
#
#  for i in range(10):
#      for idx in range(len(ds)):
#          print(i, idx)
#          #  a = ds[idx]
#          #  print(a)
