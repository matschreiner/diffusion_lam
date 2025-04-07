import os
import shutil

import numpy as np
import xarray as xr

PATH = "storage/mini.zarr"


def generate_dataset():
    if os.path.exists(PATH):
        shutil.rmtree(PATH)

    data = xr.open_zarr(
        "/dmidata/projects/cloudphysics/danra/data/v0.5.0/height_levels.zarr"
    )

    #  data = data.isel(x=slice(0, 50, 100), y=slice(0, 50, 100), time=slice(0, 100))
    data = data.isel(
        x=slice(0, 256, 2),
        y=slice(0, 256, 2),
        time=slice(0, 1000),
    )

    del data["danra_projection"]

    #  data = data.chunk({"time": 1, "x": 50, "y": 50})

    for key in data.keys():
        data[key].encoding["chunks"] = (1, 20, 20)
        data[key].encoding["preferred_chunks"] = {"time": 1, "x": 20, "y": 20}

    data.to_zarr(PATH)


generate_dataset()
