import os

# Standard library
import sys

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import xarray

import dlam


def main():
    da = xarray.open_zarr(
        "/dmidata/projects/cloudphysics/danra/data/v0.5.0/height_levels.zarr"
    )

    __import__("pdb").set_trace()  # TODO delme
    # da_stacked.dims == ('time', 'feature')
    # da_stacked.shape == (96768, 4*9*589*789)

    # 3) (optional) transpose if you want features first and time second:
    da_stacked = da_stacked.transpose("feature", "time")

    __import__("pdb").set_trace()  # TODO delme


main()
