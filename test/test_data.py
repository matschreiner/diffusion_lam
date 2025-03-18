from dlam.data import WeatherDataset

RES_PATH = "storage/danra_sample/height_levels.zarr"


#  import os
#  import shutil
#
#  import xarray as xr
#  def test_load_and_save():
#      if os.path.exists(RES_PATH):
#          shutil.rmtree(RES_PATH)
#
#      d1 = xr.open_zarr(
#          "/dmidata/projects/cloudphysics/danra/data/v0.5.0/height_levels.zarr"
#      )
#      d1 = d1.isel(x=slice(0, 10), y=slice(0, 10), time=slice(0, 100))
#
#      d1.to_zarr(RES_PATH, "w")
#      d2 = xr.open_zarr(RES_PATH)
#      assert d1["u"].values.equals(d2["u"].values())


def test_dataset_can_instantiate():
    WeatherDataset(path=RES_PATH)


def test_dataset_can_instantiate():
    WeatherDataset(path=RES_PATH)


def test_dataset_islice():
    iselection = {"x": slice(0, 5), "y": slice(0, 5), "time": slice(0, 5)}
    dataset = WeatherDataset(path=RES_PATH, iselection=iselection)
    shape = dataset.data["u"].values.shape
    assert shape[0] == 5
    assert shape[2] == 5
    assert shape[3] == 5
