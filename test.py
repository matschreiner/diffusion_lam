import numpy as np
import xarray as xr

a_values = np.arange(2, 8).reshape(2, 3) / 10
b_values = np.arange(28, 34).reshape(2, 3) / 10

ds = xr.Dataset(
    {
        "a": (["x", "y"], a_values),  # 'a' variable with dimensions ['x', 'y']
        "b": (["x", "y"], b_values),  # 'b' variable with dimensions ['x', 'y']
    },
    coords={
        "x": [1, 2],  # x coordinates
        "y": [10, 20, 30],  # y coordinates
    },
)


#  feature = xr.concat([ds["a"], ds["b"]], dim="y")
#
#  feature = feature.rename({"y": "feature_dim"})
#  feature = feature.drop("feature_dim")
#
#
#  new_ds = xr.Dataset(
#      {"feature": feature},  # Add the concatenated data as a variable in the new dataset
#  )

#  new_ds = new_ds.rename({"y": "feature_dim"})
__import__("pdb").set_trace()  # TODO delme
