import yaml

from dlam import utils
from dlam.data import WeatherDataset


def test_dataset_can_instantiate(zarr_test_path):
    WeatherDataset(path=zarr_test_path)


def test_dataset_islice(zarr_test_path):
    iselection = {"x": slice(0, 5), "y": slice(0, 5), "time": slice(0, 5)}
    dataset = WeatherDataset(path=zarr_test_path, iselection=iselection)
    assert dataset.data["u"].values.shape == (5, 9, 5, 5)


def test_dataset_slice(zarr_test_path):
    selection = {"time": slice("1990-09-01", "1990-09-03"), "altitude": [30, 100]}
    dataset = WeatherDataset(path=zarr_test_path, selection=selection)
    assert dataset.data["u"].values.shape == (24, 2, 10, 10)


def test_from_config():
    config = utils.load_yaml("test/resources/data_config.yaml")
    dataset = WeatherDataset(**config)
    assert dataset.data["u"].values.shape == (10, 9, 5, 5)


#  def test_get(zarr_test_path):
#      dataset = WeatherDataset(path=zarr_test_path)
#      d = dataset[0]
#      __import__("pdb").set_trace()  # TODO delme


def test_select_variables(zarr_test_path):
    import xarray as xr

    data = xr.open_zarr(zarr_test_path)
    del data["danra_projection"]
    data = data.isel(time=0)

    state_feature = xr.concat([data["u"], data["v"]], dim="altitude")
    data.stack(variables=["u", "v"], dim="altitude")

    print(data.u.shape)
    print(data.v.shape)
    expected_shape = (
        data["u"].shape[0] + data["v"].shape[0],
        data["v"].shape[1],
        data["v"].shape[2],
    )
    shape = state_feature.shape
    __import__("pdb").set_trace()  # TODO delme

    assert shape == expected_shape
