import time

import numpy as np
import xarray as xr
from tqdm import tqdm

#  path = "/dcai/projects/cu_0003/data/sources/danra/v0.5.0/pressure_levels.zarr"
#  original = xr.open_zarr(path).isel(time=slice(0, 10))
#  original.to_zarr("test.zarr", consolidated=True)
#
ds = xr.open_zarr("test.zarr")

fetch_times = []

print("fetching")
for i in tqdm(range(10)):
    start = time.time()
    frame = ds.isel(time=0)
    a = frame["u"].values
    fetch_times.append(time.time() - start)


print("fetch", np.mean(fetch_times))

stacking_times = []
#  print("stacking")
#  for i in range(10):
#      start = time.time()
#      ds.isel(time=0).stack(node=["x", "y"]).to_array().values
#      stacking_times.append(time.time() - start)


#  data_config = yaml.safe_load(open("test_script/config.yaml"))
#  dataset = WeatherDataset(**data_config)

#  print("stacking", np.mean(stacking_times))
