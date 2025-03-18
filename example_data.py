import os
import shutil

import numpy as np
import xarray as xr

TEST_PATH = "test/resources/example.zarr"


def test_load_and_save():
    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)

    d1 = xr.open_zarr(
        "/dmidata/projects/cloudphysics/danra/data/v0.5.0/height_levels.zarr"
    )
    d1 = d1.isel(x=slice(0, 10), y=slice(0, 10), time=slice(0, 100))

    d1.to_zarr(TEST_PATH)

    d2 = xr.open_zarr(TEST_PATH)
    assert np.all(d1["u"].values == d2["u"].values)


test_load_and_save()
